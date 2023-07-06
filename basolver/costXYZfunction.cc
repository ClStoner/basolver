#include <iostream>
#include "Sophus/sophus/se3.hpp"
#include "utility/utility.h"

#include "include/pose.h"
#include "include/costXYZfunction.h"


namespace BaSolver
{


void CostXYZfunction::ComputeResidual() {
    
    const double *param_i = verticies_[0]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi(param_i[0], param_i[1], param_i[2]);

    const double *para_w = verticies_[1]->Parameters();
    Vec3 pts_w(para_w[0], para_w[1], para_w[2]);

    Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
    Vec3 pts_camera_i = Rci[0] * pts_imu_i + tci[0];

    Eigen::Map<Eigen::Vector2d> residual(residual_);
    double dep_i = pts_camera_i.z();
    residual = (pts_camera_i / dep_i).head<2>() - obs_.head<2>();
    residual = information_ * residual;
}


void CostXYZfunction::ComputeJacobians() 
{
    ComputeOnlyJacobians();
    int index_0 = verticies_[0]->OrderingId();
    int index_1 = verticies_[1]->OrderingId();

    jacTjac(jacobians_[0], jacobians_[0], 6, 6, 2, index_0, index_0, ordering_poses_, Hpp);
    jacTjac(jacobians_[0], jacobians_[1], 6, 3, 2, index_0 - ext_size_, index_1 - ordering_poses_, ordering_landmarks_, Hpm);
    jacTjac(jacobians_[1], jacobians_[1], 3, 3, 2, index_1 - ordering_poses_, index_1 - ordering_poses_, ordering_landmarks_, Hmm);


    jacTres(jacobians_[0], residual_, 6, 2, index_0, b);
    jacTres(jacobians_[1], residual_, 3, 2, index_1, b);
}


void CostXYZfunction::ComputeOnlyJacobians()
{
    const double *param_i = verticies_[0]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi(param_i[0], param_i[1], param_i[2]);

    const double *para_w = verticies_[1]->Parameters();
    Vec3 pts_w(para_w[0], para_w[1], para_w[2]);



    Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
    Vec3 pts_camera_i = Rci[0] * pts_imu_i + tci[0];

    Eigen::Map<Eigen::Vector2d> residual(residual_);
    double dep_i = pts_camera_i.z();
    residual = (pts_camera_i / dep_i).head<2>() - obs_.head<2>();
    residual = information_ * residual;

    if(jacobians_)
    {
        Mat33 Ri = Qi.toRotationMatrix();
        Mat33 ric = Ric[0];
        Mat23 reduce(2, 3);
        reduce << 1. / dep_i, 0, -pts_camera_i(0) / (dep_i * dep_i),
            0, 1. / dep_i, -pts_camera_i(1) / (dep_i * dep_i);

        reduce = information_ * reduce;

        if(!verticies_[0]->IsFixed())
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians_[0]); 
            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * -Ri.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_i);
            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
        }
        else 
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians_[0]);
            jacobian_pose_i.setZero();
        }
        if(!verticies_[1]->IsFixed())
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_feature(jacobians_[1]);
            jacobian_feature = reduce * ric.transpose() * Ri.transpose();
        }
        else 
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians_[1]);
            jacobian_feature.setZero();
        }
    }
    ComputeChi2();
    UpdateJacRes();
    
}

}