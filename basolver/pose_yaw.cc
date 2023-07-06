#include "include/pose_yaw.h"


namespace BaSolver{

void PoseYaw::Plus(const double *delta)
{
    for(int i = 0; i < num_dimension_; i ++) parameters_[i] += delta[i];
}


}