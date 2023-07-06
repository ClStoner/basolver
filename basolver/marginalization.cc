#include "include/marginalization.h"

#include <fstream>

namespace BaSolver {

void ResidualBlockInfo::Evaluate()
{
    int num_res = edge->ResidualDimension();
    edge->ComputeOnlyJacobians();
    raw_residual = edge->Residual();
    raw_jacobians = edge->Jacobians();
}
MarginalizationInfo::~MarginalizationInfo()
{
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete[] it->second;
    for (int i = 0; i < (int)factors.size(); i++)
    {
        delete factors[i];
    }
}
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo* residual_block_info)
{
    factors.emplace_back(residual_block_info);
    std::vector<int> parameter_blocks_id = residual_block_info->parameter_blocks_id;
    std::vector<int> parameter_block_sizes = residual_block_info->parameter_blocks_size;
    for(int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i ++)
    {
        int addr = parameter_blocks_id[i];
        int size = parameter_block_sizes[i];
        parameter_block_size[addr] = size;
        ver_set.insert(addr);

    }
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        int addr = parameter_blocks_id[residual_block_info->drop_set[i]];
        drop_set.insert(addr);
    }
}
void MarginalizationInfo::preMarginalize()
{
    for(auto it : factors)
    {
        it->Evaluate();
        std::vector<int> block_sizes = it->parameter_blocks_size;
        for(int i = 0; i < static_cast<int>(block_sizes.size()); i ++)
        {
            int addr = it->parameter_blocks_id[i];
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}
int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}
int MarginalizationInfo::globalSize(int size) const{
    return size == 6 ? 7 : size;
}


void MarginalizationInfo::marginalize()
{
    int pos = 0;
    for(auto it : drop_set)
    {
        parameter_block_idx[it] = pos;
        pos += localSize(parameter_block_size[it]);
    }
    m = pos;
    for(auto it : ver_set)
    {
        if(parameter_block_idx.find(it) == parameter_block_idx.end())
        {
            parameter_block_idx[it] = pos;
            pos += localSize(parameter_block_size[it]);
        }
    }
    int stRow[pos];
    for(auto it : ver_set)
    {
        int idx = parameter_block_idx[it];
        int dim = parameter_block_size[it];
        if(dim == 7) dim = 6;
        int st = idx + dim;
        for(int i = idx; i < st; i ++)
        {
            stRow[i] = st;
        }
    }
    n = pos - m;
    TicToc t_summing;
    double *Ah;
    double *bb;
    Ah = new double[pos * pos];
    bb = new double[pos];
    memset(Ah, 0, sizeof(double) * (pos * pos));
    memset(bb, 0, sizeof(double) * (pos));

    
    for(auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[it->parameter_blocks_id[i]];
            int size_i = parameter_block_size[it->parameter_blocks_id[i]];
            if (size_i == 7)
                size_i = 6;
            double* jacobian_i = it->raw_jacobians[i];
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[it->parameter_blocks_id[j]];
                int size_j = parameter_block_size[it->parameter_blocks_id[j]];
                if (size_j == 7)
                    size_j = 6;
                double* jacobian_j = it->raw_jacobians[j];

                if(idx_i <= idx_j)
                {
                    jacTjac(jacobian_i, jacobian_j, size_i, size_j, it->edge->ResidualDimension(), idx_i, idx_j, pos, Ah);
                }
                else 
                {
                    jacTjac(jacobian_j, jacobian_i, size_j, size_i, it->edge->ResidualDimension(), idx_j, idx_i, pos, Ah);
                }
            }
            jacTres(jacobian_i, it->raw_residual, size_i, it->edge->ResidualDimension(), idx_i, bb);
        }
    }

    for(int i = 0; i < pos; i ++)
    {
        for(int j = stRow[i]; j < pos; j ++)
        {
            Ah[j * pos + i] = Ah[i * pos + j];
        }
    }
    Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor>(Ah, pos, pos);
    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(bb, pos);

    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;

    delete[] Ah;
    delete[] bb;
}
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<int, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();


    for(auto &it : ver_set)
    {
        if(parameter_block_idx[it] >= m)
        {
            keep_block_size.push_back(parameter_block_size[it]);
            keep_block_idx.push_back(parameter_block_idx[it]);
            keep_block_data.push_back(parameter_block_data[it]);
            keep_block_addr.push_back(addr_shift[it]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);
    // std::cout << "get Para size "
    return keep_block_addr;
}

void EdgePrior::ComputeResidual()
{   

    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        
        Eigen::VectorXd x = Eigen::Map<VecX>(verticies_[i]->Parameters(), size);
        Eigen::VectorXd x0 = Eigen::Map<VecX>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<VecX> residual(residual_, residual_dimension_);
    residual = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
}
void EdgePrior::ComputeJacobians()
{
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        
        Eigen::VectorXd x = Eigen::Map<VecX>(verticies_[i]->Parameters(), size);
        Eigen::VectorXd x0 = Eigen::Map<VecX>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<VecX> residual(residual_, residual_dimension_);
    residual = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
        int idx = marginalization_info->keep_block_idx[i] - m;

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jaco(jacobians_[i], residual_dimension_, local_size);
        jaco = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
    }
    ComputeChi2();
    UpdateJacRes();
    for(int i = 0; i < num_verticies_; i ++)
    {
        auto v_i = verticies_[i];
        int index_i = v_i->OrderingId();
        int dim_i = v_i->LocalDimension();
        for(int j = i; j < num_verticies_; j ++)
        {
            auto v_j = verticies_[j];
            int index_j = v_j->OrderingId();
            int dim_j =  v_j->LocalDimension();  
            jacTjac(jacobians_[i], jacobians_[j], dim_i, dim_j, residual_dimension_, index_i, index_j, ordering_poses_, Hpp);
        }
        jacTres(jacobians_[i], residual_, dim_i, residual_dimension_, index_i, b);
    }
}
void EdgePrior::ComputeOnlyJacobians()
{
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        
        Eigen::VectorXd x = Eigen::Map<VecX>(verticies_[i]->Parameters(), size);
        Eigen::VectorXd x0 = Eigen::Map<VecX>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<VecX> residual(residual_, residual_dimension_);
    residual = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
        int idx = marginalization_info->keep_block_idx[i] - m;

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jaco(jacobians_[i], residual_dimension_, local_size);
        jaco = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
    }
}


}
