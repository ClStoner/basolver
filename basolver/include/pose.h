/**
 * @file pose.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 相机位姿
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_POSE_H
#define BASOLVER_POSE_H

#include <memory>
#include "vertex.h"

namespace BaSolver {

/**
 * @brief 相机位姿节点，继承自Vertex类
 * 
 */
class Pose : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * @brief 构造函数
     * 
     */
    Pose() : Vertex(7, 6) {}

    /**
     * @brief 重载位姿加法
     * 
     * @param delta 
     */
    virtual void Plus(const double *delta) override;
    /**
     * @brief 返回节点类型
     * 
     * @return std::string 
     */
    std::string TypeInfo() const {
        return "Pose";
    }
};

}


#endif //BASOLVER_POSE_H
