#include <iostream>
#include "Sophus/sophus/se3.hpp"
#include "utility/utility.h"


#include "include/pose_yaw.h"
#include "include/costFourPoseGraphfunction.h"
#include "include/problem.h"



namespace BaSolver{

void CostFourPoseGraphFunction::ComputeResidual()
{

}
void CostFourPoseGraphFunction::ComputeJacobians()
{
    ComputeOnlyJacobians();
    int index_0 = verticies_[0]->OrderingId();
    int index_1 = verticies_[1]->OrderingId();

    jacTjac(jacobians_[0], jacobians_[0], 4, 4, 4, index_0, index_0, ordering_poses_, Hpp);
    jacTjac(jacobians_[0], jacobians_[1], 4, 4, 4, index_0, index_1, ordering_poses_, Hpp);
    jacTjac(jacobians_[1], jacobians_[1], 4, 4, 4, index_1, index_1, ordering_poses_, Hpp);

    jacTres(jacobians_[0], residual_, 4, 4, index_0, b);
    jacTres(jacobians_[1], residual_, 4, 4, index_1, b);
}
void CostFourPoseGraphFunction::ComputeOnlyJacobians()
{
    const double *param_0 = verticies_[0]->Parameters();

    double yaw_i = param_0[3];
    double ti[3] = {param_0[0], param_0[1], param_0[2]};



    const double *param_1 = verticies_[1]->Parameters();
    double yaw_j = param_1[3];
    double tj[3] = {param_1[0], param_1[1], param_1[2]};

    double tij[3];
    tij[0] = tj[0] - ti[0];
    tij[1] = tj[1] - ti[1];
    tij[2] = tj[2] - tj[2];

    double R_i[9];
    yawPitchRollToRotationMatrix(yaw_i, relative_euler_[1], relative_euler_[2], R_i);
    double inv_R_i[9];
    rotationMatrixTranpose(R_i, inv_R_i);

    double t_i_ij[3];
    rotationMatrixRotatePoint(inv_R_i, tij, t_i_ij);
    
    residual_[0] = t_i_ij[0] - relative_t_[0];
    residual_[1] = t_i_ij[1] - relative_t_[1];
    residual_[2] = t_i_ij[2] - relative_t_[2];
    residual_[3] = normalizeAngle(yaw_j - yaw_i - relative_euler_[0]);

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