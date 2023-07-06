/**
 * @file motion.h
 * @author Chenglei (ClStoner@163.com)
 * @brief IMU运动信息
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_MOTION_H
#define BASOLVER_MOTION_H

#include <memory>
#include "vertex.h"

namespace BaSolver {


/**
 * @brief IMU运动节点，继承于Vertex类
 * 
 */
class Motion : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * @brief 构造函数
     * 
     */
    Motion() : Vertex(9) {}
    /**
     * @brief 返回节点类型
     * 
     * @return std::string 
     */
    std::string TypeInfo() const {
        return "Motion";
    }

};

}


#endif //BASOLVER_MOTION_H
