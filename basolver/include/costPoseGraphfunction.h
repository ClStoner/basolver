
#ifndef BASOLVER_COSTPOSEGRAHPFUNCTION_H
#define BASOLVER_COSTPOSEGRAHPFUNCTION_H
#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"


namespace BaSolver {

class CostPoseGraphFunction : public Edge 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CostPoseGraphFunction(const Vec3 &relative_t, const Qd &relative_q)
        : Edge(6, 2, std::vector<std::string>{"Pose", "Pose"})
    {
        relative_t_ = relative_t;
        relative_q_ = relative_q;
    }
    /**
     * @brief 返回约束类型
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const override { return "CostPoseGraphFunction"; }

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
    /// @brief 两关键帧相对位姿观测值
    Qd relative_q_;
    Vec3 relative_t_;
    double t_var_;
    double q_var_;


};

}


#endif //BASOLVER_COSTPOSEGRAHPFUNCTION_H