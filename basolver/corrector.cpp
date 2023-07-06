#include "include/corrector.h"
#include <iostream>

namespace BaSolver {

    
Corrector::Corrector(double sq_norm, double *rho)
{
    sqrt_rho1_ = std::sqrt(rho[1]);
    if((sq_norm == 0.0 || rho[2] <= 0))
    {
        residual_scaling_ = sqrt_rho1_;
        alpha_sq_norm_ = 0.0;
        return;
    }
    const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
    const double alpha = 1.0 - std::sqrt(D);
    residual_scaling_ = sqrt_rho1_ / (1.0 - alpha);
    alpha_sq_norm_ = alpha / sq_norm;
}
void Corrector::CorrectResiduals(int num_rows, double* residuals)
{
    for(int i = 0; i < num_rows; i ++) residuals[i] *= residual_scaling_;
}
void Corrector::CorrectJacobian(const int num_rows,
                                const int num_cols,
                                double* residuals,
                                double* jacobian)
{
    if(alpha_sq_norm_ == 0.0)
    {
        for(int i = 0; i < num_rows * num_cols; i ++) jacobian[i] *= sqrt_rho1_;
        return;
    }
    for (int c = 0; c < num_cols; ++c) {
    double r_transpose_j = 0.0;
    for (int r = 0; r < num_rows; ++r) {
      r_transpose_j += jacobian[r * num_cols + c] * residuals[r];
    }

    for (int r = 0; r < num_rows; ++r) {
      jacobian[r * num_cols + c] = sqrt_rho1_ *
          (jacobian[r * num_cols + c] -
           alpha_sq_norm_ * residuals[r] * r_transpose_j);
    }
  }
}


}