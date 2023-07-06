#include <iostream>
#include "Sophus/sophus/se3.hpp"
#include "utility/utility.h"


#include "include/pose.h"
#include "include/costPoseGraphfunction.h"
#include "include/problem.h"

namespace BaSolver {

void CostPoseGraphFunction::ComputeResidual()
{
    /// 残差计算
    const double *param_0 = verticies_[0]->Parameters();
    Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Vec3 Pi(param_0[0], param_0[1], param_0[2]);


    const double *param_1 = verticies_[1]->Parameters();
    Qd Qj(param_1[6], param_1[3], param_1[4], param_1[5]);
    Vec3 Pj(param_1[0], param_1[1], param_1[2]);

    // Vec3 t_w_ij = Pj - Pi;

    Qd Qi_inv = Qi.inverse();
    Vec3 tij = Qi_inv * (Pj - Pi);
    
    residual_[0] = (tij[0] - relative_t_[0]) / t_var_;
    residual_[1] = (tij[1] - relative_t_[1]) / t_var_;
    residual_[2] = (tij[2] - relative_t_[2]) / t_var_;

    Qd q_error = relative_q_.inverse() * Qi_inv * Qj;
    q_error.normalized();
    residual_[3] = 2.0 * q_error.x() / q_var_;
    residual_[4] = 2.0 * q_error.y() / q_var_;
    residual_[5] = 2.0 * q_error.z() / q_var_;
}
void CostPoseGraphFunction::ComputeJacobians()
{
    ComputeOnlyJacobians();
    int index_0 = verticies_[0]->OrderingId();
    int index_1 = verticies_[1]->OrderingId();

    jacTjac(jacobians_[0], jacobians_[0], 6, 6, 6, index_0, index_0, ordering_poses_, Hpp);
    jacTjac(jacobians_[0], jacobians_[1], 6, 6, 6, index_0, index_1, ordering_poses_, Hpp);
    jacTjac(jacobians_[1], jacobians_[1], 6, 6, 6, index_1, index_1, ordering_poses_, Hpp);

    jacTres(jacobians_[0], residual_, 6, 6, index_0, b);
    jacTres(jacobians_[1], residual_, 6, 6, index_1, b);

}
void CostPoseGraphFunction::ComputeOnlyJacobians()
{
    /// 残差计算
    const double *param_0 = verticies_[0]->Parameters();
    Qd Qi(param_0[6], param_0[3], param_0[4], param_0[5]);
    Vec3 Pi(param_0[0], param_0[1], param_0[2]);


    const double *param_1 = verticies_[1]->Parameters();
    Qd Qj(param_1[6], param_1[3], param_1[4], param_1[5]);
    Vec3 Pj(param_1[0], param_1[1], param_1[2]);

    // Vec3 t_w_ij = Pj - Pi;

    Qd Qi_inv = Qi.inverse();
    Vec3 tij = Qi_inv * (Pj - Pi);
    
    residual_[0] = (tij[0] - relative_t_[0]) / t_var_;
    residual_[1] = (tij[1] - relative_t_[1]) / t_var_;
    residual_[2] = (tij[2] - relative_t_[2]) / t_var_;

    Qd q_error = relative_q_.inverse() * Qi_inv * Qj;
    q_error.normalized();
    residual_[3] = 2.0 * q_error.x() / q_var_;
    residual_[4] = 2.0 * q_error.y() / q_var_;
    residual_[5] = 2.0 * q_error.z() / q_var_;

    /*
        r_t = R_i^T * (t_j - t_i) - t_ij
        r_q = R_ij^T * R_i^T * R_j
    */

   /// 雅克比计算
    if(jacobians_)
    {
        if(!verticies_[0]->IsFixed())
        {

        }
        else 
        {

        }

        if(!verticies_[1]->IsFixed())
        {

        }
        else 
        {

        }

    }
    ComputeChi2();
    UpdateJacRes();
}


}