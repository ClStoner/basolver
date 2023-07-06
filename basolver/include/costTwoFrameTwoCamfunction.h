/**
 * @file costTwoFrameTwoCamfunction.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 双目视觉重投影约束，路标点被一帧左目观测到，同时被另外一帧右目观测到
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef BASOLVER_TWOFRAMETWOCAMFUNCTION_H
#define BASOLVER_TWOFRAMETWOCAMFUNCTION_H
#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"


namespace BaSolver
{
/**
 * @brief 此约束为双目视觉重投影约束类,表示路标点被主导帧左目观测到以及观测帧右目观测到，为三元边，与之相关的节点有：
 *          路标点的逆深度、主导帧位姿、当前帧位姿。
 *          继承自Edge类
 *          节点添加的顺序必须为：主导帧位姿、观测帧位姿、路标点逆深度。
 * 
 */
class CostTwoFrameTwoCamFunction : public Edge
{
public: 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * @brief 构造函数
     * 
     * @param pts_i 3D点在第一帧左目下的归一化坐标
     * @param pts_j 3D点在另一观测帧右目下的归一化坐标
     */
    CostTwoFrameTwoCamFunction(const Vec3 &pts_i, const Vec3 &pts_j)
        : Edge(2, 3, std::vector<std::string>{"Pose", "Pose", "FeatureMeasure"})
    {
        pts_i_ = pts_i;
        pts_j_ = pts_j;
    }
    /**
     * @brief 返回约束类型
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const override {return "CostTwoFrameTwoCamFunction"; }
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
    /// @brief 路标点在对应帧中左右目下的归一化坐标
    Eigen::Vector3d pts_i_, pts_j_;

};


}
#endif ///BASOLVER_TWOFRAMETWOCAMFUNCTION_H