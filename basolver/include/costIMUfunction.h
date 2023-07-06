/**
 * @file costIMUfunction.h
 * @author Chenglei (ClStoner@163.com)
 * @brief IMU约束类
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_COSTIMUFUNCTION_H
#define BASOLVER_COSTIMUFUNCTION_H

#include <memory>
#include <string>
#include "../Sophus/sophus/se3.hpp"

#include "eigen_types.h"
#include "edge.h"
#include "integration_base.h"
#include "parameters.h"
namespace BaSolver {

/**
 * @brief 此约束为IMU约束，为四元约束，与之相连的节点有：上一帧位姿、IMU运动信息、当前帧位姿、IMU运动信息
 *          继承自Edge类
 *          节点顺序必须为：上一帧位姿、当前帧位姿、IMU运动信息、IMU运动信息
 */
class CostIMUFunction : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @brief 构造函数
     * 
     * @param _pre_integration 相邻帧之间IMU预积分信息
     */
    explicit CostIMUFunction(IntegrationBase* _pre_integration):pre_integration_(_pre_integration),
          Edge(15, 4, std::vector<std::string>{"Pose", "Pose", "Motion", "Motion"}) {

    }

    /**
     * @brief 返回边的类型信息
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const override { return "CostIMUFunction"; }

    /**
     * @brief 计算残差
     * 
     */
    virtual void ComputeResidual() override;

    /**
     * @brief 计算雅可比
     * 
     */
    virtual void ComputeJacobians() override;

    virtual void ComputeOnlyJacobians() override;
private:
    enum StateOrder
    {
        O_P = 0,
        O_R = 3,
        O_V = 6,
        O_BA = 9,
        O_BG = 12
    };
    /// @brief IMU预积分信息
    IntegrationBase* pre_integration_;
    /// @brief 重力加速度
    static Vec3 gravity_;
    /// @brief IMU预积分相关矩阵块
    Mat33 dp_dba_ = Mat33::Zero();
    Mat33 dp_dbg_ = Mat33::Zero();
    Mat33 dr_dbg_ = Mat33::Zero();
    Mat33 dv_dba_ = Mat33::Zero();
    Mat33 dv_dbg_ = Mat33::Zero();
};

}
#endif //BASOLVER_COSTIMUFUNCTION_H
