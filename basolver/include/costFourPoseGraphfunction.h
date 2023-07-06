#ifndef BASOLVER_COSTFOURPOSEGRAPHFUNCTION_H
#define BASOLVER_COSTFOURPOSEGRAPHFUNCTION_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"

namespace BaSolver{

class CostFourPoseGraphFunction : public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CostFourPoseGraphFunction(const Vec3 &relative_t, const Vec3 &relative_euler)
        : Edge(4, 2, std::vector<std::string>{"PoseYaw", "PoseYaw"})
        {
            relative_t_ = relative_t;
            relative_euler_ = relative_euler;
        }

     /**
     * @brief 返回约束类型
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const override { return "CostFourPoseGraphFunction"; }

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
    
    Vec3 relative_t_;
    Vec3 relative_euler_;

};


}

#endif