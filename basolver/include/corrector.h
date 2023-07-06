/**
 * @file corrector.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 利用核函数对雅克比和残差进行矫正
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef BASOLVER_CORRECTOR_H
#define BASOLVER_CORRECTOR_H


#include <memory>
#include <string>
#include "eigen_types.h"
#include <eigen3/Eigen/Dense>
#include "loss_function.h"


namespace BaSolver
{
/**
 * @brief 矫正类，使用核函数矫正残差雅克比
 */
class Corrector
{
public:
    /**
     * @brief 构造函数
     * 
     * @param sq_norm 残差的平方求和
     * @param rho 对残差求导 rho[0] : 残差, rho[1] : 一阶导, rho[2] ; 二阶导 
     */
    Corrector(double sq_norm, double *rho);
    /**
     * @brief 矫正残差
     * 
     * @param residual 残差
     */
    void CorrectResiduals(int num_rows, double* residuals);

    /**
     * @brief 矫正雅克比
     * 
     * @param jacobian 雅克比
     * @param residual 残差
     */
    void CorrectJacobian(const int num_rows,
                                const int num_cols,
                                double* residuals,
                                double* jacobian);
private:
    /// @brief 残差一阶导算数平方根
    double sqrt_rho1_;
    /// @brief 残差尺度
    double residual_scaling_;
    /// @brief 用于雅克比矫正
    double alpha_sq_norm_;  
};

}



#endif BASOLVER_CORRECTOR_H
