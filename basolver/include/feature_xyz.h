/**
 * @file feature_xyz.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 以xyz形式存储路标点
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_FEATURE_XYZ_H
#define BASOLVER_FEATURE_XYZ_H

#include "vertex.h"

namespace BaSolver {

/**
 * @brief 以xyz形式存储路标点，继承与Vertex类
 */
class FeatureMeasureXYZ : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * @brief 构造函数
     * 
     */
    FeatureMeasureXYZ() : Vertex(3) {}
    /**
     * @brief 返回节点类型
     * 
     * @return std::string 
     */
    std::string TypeInfo() const { return "FeatureMeasureXYZ"; }
};

}


#endif ///BASOLVER_FEATURE_XYZ_H
