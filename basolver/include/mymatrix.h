#ifndef BASOLVER_MYMATRIX_H
#define BASOLVER_MYMATRIX_H

#include <iostream>
#include "parameters.h"
#include <cmath>
namespace BaSolver
{


/**
 * @brief 将3D坐标进行[R,t]变换，返回变换后在坐标为res
 * 
 * @param R 旋转矩阵 
 * @param t 平移向量
 * @param p 3D坐标
 * @param res 返回投影坐标
 */
void tranlate(const double *R, const double *t, const double *p, double *res);
/**
 * @brief 将3D点坐标进行【R,t】的逆变换，返回变换后在坐标为res
 * 
 * @param R 
 * @param t 
 * @param p 
 * @param res 
 */
void invTranlate(const double *R, const double *t, double *p, double *res);
/**
 * @brief 雅克比矩阵相乘 A^TB
 * 
 * @param A A(p * n)
 * @param B B(p * m)
 * @param n 
 * @param m 
 * @param p 
 * @param res res(n * m)
 */
void jacTjac(const double *A, const double *B, const int n, const int m, const int p, double *res);
/**
 * @brief 雅克比矩阵相乘 A^TB，结果填充到信息矩阵H对应块
 * 
 * @param A A(p * n)
 * @param B B(p * m)
 * @param n 
 * @param m 
 * @param p 
 * @param idx_i 信息矩阵左上角
 * @param idx_j 
 * @param sz 填充矩阵列的维度
 * @param res 信息矩阵 res.block(idx_i, idx_j, n, m) = A^TB
 */
void jacTjac(const double *A, const double *B, const int n, const int m, const int p, const int idx_i, const int idx_j, const int sz, double *res);
/**
 * @brief 雅克比与残差相乘
 * 
 * @param A A(p * n)
 * @param R R(p * 1)
 * @param n 
 * @param p 
 * @param res res(n * 1) = -A^T * R
 */
void jacTres(const double *A, const double *R, const int n, const int p, double *res);
/**
 * @brief 雅克比与残差相乘，结果填充到b对应块
 * 
 * @param A A(p * n)
 * @param R R(p * 1)
 * @param n 
 * @param p 
 * @param idx 在b中的起始位置
 * @param res res.segment(idx, n) = -A^T * R
 */
void jacTres(const double *A, const double *R, const int n, const int p, const int idx, double *res);

/**
 * @brief 信息矩阵对应块减半 H.block(idx_i, idx_j, n, m) /= 2.0
 * 
 * @param n 对应块维度(n*m)
 * @param m 
 * @param idx_i 
 * @param idx_j 
 * @param sz 
 * @param H 
 */
void hessianDviTwo(const int n, const int m, const int idx_i, const int idx_j, const int sz, double *H);
/**
 * @brief 矩阵相乘
 * 
 * @param A A(n * p)
 * @param B B(p * m)
 * @param n 
 * @param m 
 * @param p 
 * @param res res(n * m)
 */
void multiMatrix(const double *A, const double *B, const int n, const int m, const int p, double *res);
/**
 * @brief 矩阵转置
 * 
 * @param A A(n * m)
 * @param n 
 * @param m 
 * @param res res(m * n)
 */
void transpose(const double *A, const int n, const int m, double *res);
/**
 * @brief 打印矩阵
 * 
 * @param A A(n * m)
 * @param n 
 * @param m 
 */
void printMatrix(const double *A, const int n, const int m);
/**
 * @brief 数乘矩阵
 * 
 * @param A A = A * x
 * @param n 
 * @param m 
 * @param x 
 */
void numMulMatrix(double *A, const int n, const int m, double x);
/**
 * @brief 矩阵相加
 * 
 * @param A A(n * m)
 * @param B B(n * m)
 * @param n 
 * @param m 
 * @param C C = A + B 
 */
void addMatrix(const double *A, const double *B, const int n, const int m, double *C);

/**
 * @brief 计算分子模型 modelcosechange = 0.5 * (delta_x.dot(currentLambda * delta_x + b))
 * 
 * @param delta 
 * @param b 
 * @param lambda 
 * @param sz 
 * @return double 
 */
double computeModelChange(const double *delta, const double *b, const double lambda, const int sz);

/**
 * @brief 从矩阵A中取小块矩阵块 B = A.block(index_x, index_y, sx, sy)
 * 
 * @param A 
 * @param index_x 
 * @param index_y 
 * @param sx 
 * @param sy 
 * @param sz 
 * @param B 
 */
void block(const double *A, const int index_x, const int index_y, const int sx, const int sy, const int sz, double *b);

/**
 * @brief 设置矩阵A中对应块为B A.block(index_x, index_y, sx, sy) = B
 * 
 * @param A 
 * @param index_x 
 * @param index_y 
 * @param sx 
 * @param sy 
 * @param sz 
 * @param b 
 */

void setBlock(double *A, const int index_x, const int index_y, const int sx, const int sy, const int sz, const double *b);


void setMultiBlock(const double *A, const double *B, const int index_x, const int index_y, const int sz, double *C);

/**
 * @brief 将欧拉角yaw,pitch,roll转为旋转矩阵
 * @param yaw
 * @param pitch
 * @param roll
 * @param R[9]
*/
void yawPitchRollToRotationMatrix(const double yaw, const double pitch, const double roll, double R[9]);

/**
 * @brief 角度归一化
 * @param angle_degrees
 * @return 返回归一化角度
*/
double normalizeAngle(const double &angle_degrees);
/**
 * @brief 获取旋转矩阵的逆
 * @param R 旋转矩阵
 * @param inv_R 旋转矩阵的逆
*/
void rotationMatrixTranpose(const double R[9], double inv_R[9]);
/**
 * @brief 3D 点经过旋转矩阵完成坐标变换
 * @param R 旋转矩阵
 * @param t 3D 点坐标
 * @param r_t 变换后坐标
*/
void rotationMatrixRotatePoint(const double R[9], const double t[3], double r_t[3]);


}



#endif