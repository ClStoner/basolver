#include "include/pose.h"
#include "Sophus/sophus/se3.hpp"
//#include <iostream>
namespace BaSolver {

void Pose::Plus(const double *delta) {
    // 更新平移
    parameters_[0] += delta[0];
    parameters_[1] += delta[1];
    parameters_[2] += delta[2];
    //右乘更新旋转
    Qd q(parameters_[6], parameters_[3], parameters_[4], parameters_[5]);
    q = q * Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion();  //李代数表示旋转
    q.normalized();
    parameters_[3] = q.x();
    parameters_[4] = q.y();
    parameters_[5] = q.z();
    parameters_[6] = q.w();
}


}
