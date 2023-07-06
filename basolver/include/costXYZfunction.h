/**
 * @file costXYZfunction.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 视觉重投影约束
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_COSTXYZFUNCTION_H
#define BASOLVER_COSTXYZFUNCTION_H


#include <memory>
#include <string>

#include <Eigen/Dense>
// #include "parameter.h"
#include "eigen_types.h"
#include "edge.h"

namespace BaSolver
{
/**
 * @brief 此约束为视觉重投影约束，为二元约束，与之相关的节点有路标点的世界坐标系下的3D坐标，观测帧位姿
 *          继承自Edge类
 *          节点顺序必须为: 观测帧位姿(body系到world系下的坐标变换)、 路标点的世界坐标系下的3D坐标
 * 
 */
class CostXYZfunction : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * @brief 构造函数
     * 
     * @param pts_i 路标点在观测帧中的归一化坐标
     */
    CostXYZfunction(const Vec3 &pts_i)
        : Edge(2, 2, std::vector<std::string>{"VertexPose", "FeatureMeasureXYZ"}) {
        obs_ = pts_i;
    }

    /**
     * @brief 返回边的类型信息
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const override { return "CostXYZfunction"; }

    /**
     * @brief 计算残差
     */
    virtual void ComputeResidual() override;
    /**
     * @brief 计算雅克比
     */
    virtual void ComputeJacobians() override;
    virtual void ComputeOnlyJacobians() override;

private:
    /// @brief 观测值
    Vec3 obs_;
};
}









#endif /// BASOLVER_COSTXYZFUNCTION_H