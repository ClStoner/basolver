/**
 * @file costfunction.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 视觉重投影约束
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_COSTFUNCTION_H
#define BASOLVER_COSTFUNCTION_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"

namespace BaSolver{

/**
 * @brief 此约束为视觉重投影约束类，为三元边，与之相关的节点有：
 *          路标点的逆深度、主导帧位姿、当前帧位姿。
 *          继承自Edge类
 *          节点添加的顺序必须为：主导帧位姿、当前帧位姿、路标点逆深度。
 * 
 */
class CostFunction : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * @brief 构造函数
     * 
     * @param pts_i 3D点在主导帧相机下的归一化坐标
     * @param pts_j 3D点在观测帧相机下的归一化坐标
     */
    CostFunction(const Vec3 &pts_i, const Vec3 &pts_j)
        : Edge(2, 3, std::vector<std::string>{"Pose", "Pose", "FeatureMeasure"}) {
        pts_i_ = pts_i;
        pts_j_ = pts_j;
    }

    /**
     * @brief 返回约束类型
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const override { return "CostFunction"; }

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
//    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
    /// @brief 路标点在对应帧中归一化坐标
    Vec3 pts_i_, pts_j_;
};

}

#endif //BASOLVER_COSTFUNCTION_H
