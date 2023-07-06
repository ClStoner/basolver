/**
 * @file marginalization.h
 * @author your name (you@domain.com)
 * @brief 边缘化相关类，来自VINS-MONO中边缘的代码
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_MARGINALIZATION_H
#define BASOLVER_MARGINALIZATION_H

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <memory>
#include <string>
#include <eigen3/Eigen/Dense>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "corrector.h"
#include "edge.h"

#include "vertex.h"

#include "parameters.h"
#include "integration_base.h"
#include "eigen_types.h"
#include "loss_function.h"

#include "mymatrix.h"

namespace BaSolver {

#pragma pack(1)
struct ResidualBlockInfo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    ResidualBlockInfo(std::shared_ptr<Edge> _edge, std::vector<int> _drop_set)
        : edge(_edge), drop_set(_drop_set) {
            parameter_blocks.clear();
            parameter_blocks_id.clear();
            parameter_blocks_size.clear();
            auto ver = edge->Verticies();
            for(int i = 0; i < ver.size(); i ++)
            {
                // parameter_blocks.push_back(ver[i]->Parameters());
                double *pa = ver[i]->Parameters();
                int sz = ver[i]->Dimension();
                parameter_blocks_size.push_back(sz);
            
                parameter_blocks.push_back(pa);
                parameter_blocks_id.push_back(ver[i]->Id());
            }
        }

    void Evaluate();//计算了雅克比和残差，并进行核函数矫正

    std::vector<double *> parameter_blocks;
    std::vector<int> parameter_blocks_size;
    std::vector<int> parameter_blocks_id;
    std::shared_ptr<Edge> edge;
    std::vector<int> drop_set;

    double **raw_jacobians;
    double *raw_residual;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

#pragma pack()
class MarginalizationInfo
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo* residual_block_info);
    void preMarginalize();
    void marginalize();

    std::vector<double *> getParameterBlocks(std::unordered_map<int, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;
    int m, n;
    std::unordered_map<int, int> parameter_block_size; //参数块大小
    int sum_block_size;
    std::unordered_map<int, int> parameter_block_idx; //参数块ID
    std::unordered_map<int, double *> parameter_block_data;//参数块数据
    std::set<long> drop_set;//边缘化节点的集合
    std::set<long> ver_set; //所有节点集合
    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};

class EdgePrior : public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // EdgePrior();
    explicit EdgePrior(MarginalizationInfo* _marginalization_info, int residual_dimension, int num_vertices, std::vector<std::string> &verticies_types) 
        : Edge(residual_dimension, num_vertices, verticies_types)
    {
        marginalization_info = _marginalization_info;
    }
    virtual std::string TypeInfo() const override {return "EdgePrior"; }
    virtual void ComputeResidual() override;
    virtual void ComputeJacobians() override;
    virtual void ComputeOnlyJacobians() override;
private:
    MarginalizationInfo* marginalization_info;
};

}



#endif