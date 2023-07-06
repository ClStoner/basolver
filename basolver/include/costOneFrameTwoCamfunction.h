/**
 * @file costOneFrameTwoCamfunction.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 双目视觉重投影约束，路标点被同一帧的左右目观测到
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_ONEFRAMETWOCAMFUNCTION_H
#define BASOLVER_ONEFRAMETWOCAMFUNCTION_H
#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"
#include "mymatrix.h"

namespace BaSolver
{
/**
 * @brief 此约束为双目视觉重投影约束类，表示一个路标点被同一帧的左右目图像观测到，为一元边，与之相关的节点有：
 *          路标点的逆深度
 *          继承自Edge类
 *          节点添加的顺序必须为：路标点逆深度。
 * 
 */
class CostOneFrameTwoCamFunction : public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * @brief 构造函数
     * 
     * @param pts_i 3D点在当前帧左目下的归一化坐标
     * @param pts_j 3D点在当前帧右目下的归一化坐标
     */
    CostOneFrameTwoCamFunction(const Vec3 &pts_i, const Vec3 &pts_j)
        : Edge(2, 1, std::vector<std::string>{"FeatureMeasure"})
    {
        pts_i_ = pts_i;
        pts_j_ = pts_j;
    }
    /**
     * @brief 返回约束类型
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const override {return "CostOneFrameTwoCamFunction"; }
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
    /// @brief 路标点在当前帧左右目下的归一化坐标
    Eigen::Vector3d pts_i_, pts_j_;

};

}

#endif ///BASOLVER_ONEFRAMETWOCAMFUNCTION_H