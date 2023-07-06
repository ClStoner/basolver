/**
 * @file feature_inverse_depth.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 以逆深度的形式存储路标点
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_FEATURE_INVERSE_DEPTH_H
#define BASOLVER_FEATURE_INVERSE_DEPTH_H

#include "vertex.h"

namespace BaSolver {

/**
 * @brief 以逆深度形式存储的路标点，继承与Vertex类
 * 
 */
class FeatureMeasure : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * @brief 构造函数
     * 
     */
    FeatureMeasure() : Vertex(1) {}
    /**
     * @brief 返回路节点类型
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const { return "FeatureMeasure"; }
};


}

#endif //BASOLVER_FEATURE_INVERSE_DEPTH_H
