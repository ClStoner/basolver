#include <iostream>
#include "Sophus/sophus/se3.hpp"
#include "utility/utility.h"

#include "include/pose.h"
#include "include/costOneFrameTwoCamfunction.h"

namespace BaSolver
{
void CostOneFrameTwoCamFunction::ComputeResidual()
{
    double inv_dep_i = verticies_[0]->Parameters()[0];
    Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
    Eigen::Vector3d pts_camera_j = R_rc_lc * pts_camera_i + t_rc_lc;
    double dep_j = pts_camera_j.z();

    Eigen::Map<Eigen::Vector2d> residual(residual_);
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
    residual = information_ * residual;
}
void CostOneFrameTwoCamFunction::ComputeJacobians()
{
    ComputeOnlyJacobians();
    int index = verticies_[0]->OrderingId();
    jacTjac(jacobians_[0], jacobians_[0], 1, 1, 2, index - ordering_poses_, index - ordering_poses_, ordering_landmarks_, Hmm);
    jacTres(jacobians_[0], residual_, 1, 2, index, b);
}
void CostOneFrameTwoCamFunction::ComputeOnlyJacobians()
{
    double inv_dep_i = verticies_[0]->Parameters()[0];
    Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
    Eigen::Vector3d pts_camera_j = R_rc_lc * pts_camera_i + t_rc_lc;
    double dep_j = pts_camera_j.z();

    Eigen::Map<Eigen::Vector2d> residual(residual_);
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
    residual = information_ * residual;

    if(jacobians_)
    {
        Eigen::Matrix<double, 2, 3> reduce(2, 3);

        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
                
        reduce = information_ * reduce;

        if(!verticies_[0]->IsFixed())
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians_[0]);
            jacobian_feature = reduce * R_rc_lc * pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);
        }
        else
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians_[0]);
            jacobian_feature.setZero();
        }
    }
    ComputeChi2();
    UpdateJacRes();
}

}