#include <iostream>
#include "Sophus/sophus/se3.hpp"
#include "utility/utility.h"

#include "include/pose.h"
#include "include/costTwoFrameTwoCamfunction.h"

namespace BaSolver
{

void CostTwoFrameTwoCamFunction::ComputeResidual()
{
    double inv_dep_i = verticies_[2]->Parameters()[0];
    Vec3 pts_camera_i = pts_i_ / inv_dep_i;
    int ps_i = vertexToPose_[verticies_[0]->Id()];
    int ps_j = vertexToPose_[verticies_[1]->Id()];
    Vec3 pts_camera_j = R_rcj_lci[ps_i][ps_j] * pts_camera_i + t_rcj_lci[ps_i][ps_j];

    Eigen::Map<Eigen::Vector2d> residual(residual_);
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();   /// J^t * J * delta_x = - J^t * r
    residual = information_ * residual; 
}
void CostTwoFrameTwoCamFunction::ComputeJacobians()
{
    ComputeOnlyJacobians();
    int index_0 = verticies_[0]->OrderingId();
    int index_1 = verticies_[1]->OrderingId();
    int index_2 = verticies_[2]->OrderingId();
    jacTjac(jacobians_[0], jacobians_[0], 6, 6, 2, index_0, index_0, ordering_poses_, Hpp);
    jacTjac(jacobians_[0], jacobians_[1], 6, 6, 2, index_0, index_1, ordering_poses_, Hpp);
    jacTjac(jacobians_[1], jacobians_[1], 6, 6, 2, index_1, index_1, ordering_poses_, Hpp);
    
    jacTjac(jacobians_[0], jacobians_[2], 6, 1, 2, index_0 - ext_size_, index_2 - ordering_poses_, ordering_landmarks_, Hpm);
    jacTjac(jacobians_[1], jacobians_[2], 6, 1, 2, index_1 - ext_size_, index_2 - ordering_poses_, ordering_landmarks_, Hpm);
    jacTjac(jacobians_[2], jacobians_[2], 1, 1, 2, index_2 - ordering_poses_, index_2 - ordering_poses_, ordering_landmarks_, Hmm);

    jacTres(jacobians_[0], residual_, 6, 2, index_0, b);
    jacTres(jacobians_[1], residual_, 6, 2, index_1, b);
    jacTres(jacobians_[2], residual_, 1, 2, index_2, b);
}
void CostTwoFrameTwoCamFunction::ComputeOnlyJacobians()
{
    double inv_dep_i = verticies_[2]->Parameters()[0];
    Vec3 pts_camera_i = pts_i_ / inv_dep_i;
    int ps_i = vertexToPose_[verticies_[0]->Id()];
    int ps_j = vertexToPose_[verticies_[1]->Id()];
    Vec3 pts_camera_j = R_rcj_lci[ps_i][ps_j] * pts_camera_i + t_rcj_lci[ps_i][ps_j];

    Eigen::Map<Eigen::Vector2d> residual(residual_);
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();   /// J^t * J * delta_x = - J^t * r
    residual = information_ * residual; 
    

    if(jacobians_)
    {        
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
        reduce = information_ * reduce;

        if(!verticies_[0]->IsFixed())
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor >> jacobian_pose_i(jacobians_[0]);
            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = R_rcj_w[ps_i][ps_j];
            jaco_i.rightCols<3>() = R_rcj_bi[ps_i][ps_j] * -Utility::skewSymmetric(Ric[0] * pts_camera_i + tic[0]);
            jacobian_pose_i = reduce * jaco_i;
        }
        else
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians_[0]);
            jacobian_pose_i.setZero();
        }
            
        if(!verticies_[1]->IsFixed())
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6,Eigen::RowMajor>> jacobian_pose_j(jacobians_[1]);
            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = -R_rcj_w[ps_i][ps_j];
            jaco_j.rightCols<3>() = Rci[1] * Utility::skewSymmetric(R_bj_ci[ps_i][ps_j] * pts_camera_i + t_bj_ci[ps_i][ps_j]);
            jacobian_pose_j = reduce * jaco_j;
        }
        else 
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians_[1]);
            jacobian_pose_j.setZero();
        }
    
        if(!verticies_[2]->IsFixed())
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians_[2]);
            jacobian_feature = reduce * R_rcj_lci[ps_i][ps_j] * pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);
        }
        else
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians_[2]);
            jacobian_feature.setZero();
        }
    }
    ComputeChi2();
    UpdateJacRes();
}
}