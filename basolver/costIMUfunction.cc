#include "basolver.h"
#include <iostream>

namespace BaSolver {
using Sophus::SO3d;

Vec3 CostIMUFunction::gravity_ = Vec3(0, 0, 9.8);

void CostIMUFunction::ComputeResidual() {

    const double *param_0 = verticies_[0]->Parameters();
    Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Vec3 Pi(param_0[0], param_0[1], param_0[2]);

    const double *param_2 = verticies_[1]->Parameters();
    Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
    Vec3 Pj(param_2[0], param_2[1], param_2[2]);

    const double *param_1 = verticies_[2]->Parameters();
    Vec3 Vi(param_1[0], param_1[1], param_1[2]);
    Vec3 Bai(param_1[3], param_1[4], param_1[5]);
    Vec3 Bgi(param_1[6], param_1[7], param_1[8]);


    const double *param_3 = verticies_[3]->Parameters();
    Vec3 Vj(param_3[0], param_3[1], param_3[2]);
    Vec3 Baj(param_3[3], param_3[4], param_3[5]);
    Vec3 Bgj(param_3[6], param_3[7], param_3[8]);

    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residual_);
    residual = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi,
                              Pj, Qj, Vj, Baj, Bgj);
    Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance.inverse()).matrixL().transpose();
    SetInformation(sqrt_info);
    residual = sqrt_info * residual;
}

void CostIMUFunction::ComputeJacobians() {
    ComputeOnlyJacobians();
    for(int i = 0; i < 4; i ++)
    {
        auto v_i = verticies_[i];
        int index_i = v_i->OrderingId();
        int dim_i = v_i->LocalDimension();
        for(int j = i; j < 4; j ++)
        {
            auto v_j = verticies_[j];
            int index_j = v_j->OrderingId();
            int dim_j =  v_j->LocalDimension();            
            jacTjac(jacobians_[i], jacobians_[j], dim_i, dim_j, 15, index_i, index_j, ordering_poses_, Hpp);
        }
        jacTres(jacobians_[i], residual_, dim_i, 15, index_i, b);
    }
}
void CostIMUFunction::ComputeOnlyJacobians()
{
    
    const double *param_0 = verticies_[0]->Parameters();
    Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Vec3 Pi(param_0[0], param_0[1], param_0[2]);

    const double *param_2 = verticies_[1]->Parameters();
    Qd Qj(param_2[6], param_2[3], param_2[4], param_2[5]);
    Vec3 Pj(param_2[0], param_2[1], param_2[2]);

    const double *param_1 = verticies_[2]->Parameters();
    Vec3 Vi(param_1[0], param_1[1], param_1[2]);
    Vec3 Bai(param_1[3], param_1[4], param_1[5]);
    Vec3 Bgi(param_1[6], param_1[7], param_1[8]);


    const double *param_3 = verticies_[3]->Parameters();
    Vec3 Vj(param_3[0], param_3[1], param_3[2]);
    Vec3 Baj(param_3[3], param_3[4], param_3[5]);
    Vec3 Bgj(param_3[6], param_3[7], param_3[8]);

    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residual_);
    residual = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi,
                              Pj, Qj, Vj, Baj, Bgj);
    Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance.inverse()).matrixL().transpose();
    SetInformation(sqrt_info);
    residual = sqrt_info * residual;


//     std::cout << "imu residual : \n" << residual << std::endl;
    if(jacobians_)
    {
        double sum_dt = pre_integration_->sum_dt;
        Eigen::Matrix3d dp_dba = pre_integration_->jacobian.template block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = pre_integration_->jacobian.template block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = pre_integration_->jacobian.template block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = pre_integration_->jacobian.template block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = pre_integration_->jacobian.template block<3, 3>(O_V, O_BG);

        if (pre_integration_->jacobian.maxCoeff() > 1e8 || pre_integration_->jacobian.minCoeff() < -1e8)
        {
                // ROS_WARN("numerical unstable in preintegration");
        }

        if (!verticies_[0]->IsFixed())
        {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians_[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

        #if 0
                jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
        #else
                Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
                jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
        #endif

                jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));
        jacobian_pose_i = information_ * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                //     ROS_WARN("numerical unstable in preintegration");
                }
        }
       
        if (!verticies_[1]->IsFixed())
        {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians_[1]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();
        #if 0
                jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
        #else
                Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));
                jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
        #endif

                jacobian_pose_j = information_ * jacobian_pose_j;

        }
         if (!verticies_[2]->IsFixed())
        {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians_[2]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

        #if 0
                jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
        #else
                //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration_->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
        #endif

                jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = information_ * jacobian_speedbias_i;
        }
        if (!verticies_[3]->IsFixed())
        {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians_[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

                jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j = information_ * jacobian_speedbias_j;
        }

    }
    ComputeChi2();
    UpdateJacRes();
}
}
/*

jacTjac(jacobians_[0], jacobians_[0], 6, 6, 2, index_0, index_0, ordering_poses_, Hpp);
*/
