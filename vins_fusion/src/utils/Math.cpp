#include "utils/Math.h"

namespace utility
{

// transform feature
Eigen::Vector3d transformPoint(const Eigen::Vector3d &point_w, const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw)
{
    Eigen::Vector3d point_c;
    point_c = Rcw * point_w + tcw;

    return point_c;
}

// triangulate feuature
Eigen::Vector3d triangulatePoint(const Eigen::Matrix3d &leftR, const Eigen::Vector3d &leftP, const Eigen::Vector3d &leftPoint,
                                 const Eigen::Matrix3d &rightR, const Eigen::Vector3d &rightP, const Eigen::Vector3d &rightPoint)
{
    // leftR and leftP: left camera to world
    // rightR and rightP: right camera to world
    // leftPose: world to camera
    Eigen::Matrix<double, 3, 4> leftPose, rightPose;
    leftPose.leftCols<3>() = leftR.transpose();
    leftPose.rightCols<1>() = -leftR.transpose() * leftP;
    rightPose.leftCols<3>() = rightR.transpose();
    rightPose.rightCols<1>() = -rightR.transpose() * rightP;

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = leftPoint[0] * leftPose.row(2) - leftPose.row(0);
    design_matrix.row(1) = leftPoint[1] * leftPose.row(2) - leftPose.row(1);
    design_matrix.row(2) = rightPoint[0] * rightPose.row(2) - rightPose.row(0);
    design_matrix.row(3) = rightPoint[1] * rightPose.row(2) - rightPose.row(1);
    Eigen::Vector4d triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    // get p3D in world coordinate
    Eigen::Vector3d p3D;
    p3D[0] = triangulated_point(0) / triangulated_point(3);
    p3D[1] = triangulated_point(1) / triangulated_point(3);
    p3D[2] = triangulated_point(2) / triangulated_point(3);

    // transform to camera coodinate
    Eigen::Vector3d localPoint = leftPose.leftCols(3) * p3D + leftPose.rightCols(1);

    return localPoint;
}

Eigen::Vector3d triangulatePoint(const Eigen::Vector3d &leftPoint, const Eigen::Vector3d &rightPoint, const double baseLine)
{
    // leftR and leftP: camera to world
    // leftPose: world to camera
    Eigen::Matrix<double, 3, 4> leftPose, rightPose;
    leftPose.leftCols<3>() = Eigen::Matrix3d::Identity();
    leftPose.rightCols<1>() = Eigen::Vector3d::Zero();
    rightPose.leftCols<3>() = Eigen::Matrix3d::Identity();
    rightPose.rightCols<1>() = -Eigen::Vector3d{baseLine, 0, 0};

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = leftPoint[0] * leftPose.row(2) - leftPose.row(0);
    design_matrix.row(1) = leftPoint[1] * leftPose.row(2) - leftPose.row(1);
    design_matrix.row(2) = rightPoint[0] * rightPose.row(2) - rightPose.row(0);
    design_matrix.row(3) = rightPoint[1] * rightPose.row(2) - rightPose.row(1);
    Eigen::Vector4d triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    // get p3D in world coordinate
    Eigen::Vector3d p3D;
    p3D[0] = triangulated_point(0) / triangulated_point(3);
    p3D[1] = triangulated_point(1) / triangulated_point(3);
    p3D[2] = triangulated_point(2) / triangulated_point(3);

    return p3D;
}

// from vins
Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = utility::R2ypr(R0).x();
    R0 = utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;

    return R0;
}

Eigen::MatrixXd tangentBasis(Eigen::Vector3d &g0)
{
    Eigen::Vector3d b, c;
    Eigen::Vector3d a = g0.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if (a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    Eigen::MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;

    return bc;
}

}  // namespace utility