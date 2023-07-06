/**
 * @file parameters.h
 * @author ClStoner (ClStoner@163.com)
 * @brief 
 * @version 0.1
 * @date 2022-11-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef BASOLVER_PARAMETERS_H
#define BASOLVER_PARAMETERS_H

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <mutex>
#include "eigen_types.h"

namespace BaSolver
{




/// @brief 相机焦距
const double FOCAL_LENGTH = 460.0;
/// @brief 信息矩阵
const Eigen::Matrix2d project_sqrt_info_ = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();
/// @brief IMU预积分相关参数
extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;
extern int NUM_OF_CAM;
extern int NUM_OF_THREADS;
/// @brief 重力加速度
extern Eigen::Vector3d G;

extern bool STEREO;
extern double *b;
/// @brief 信息矩阵左上角
extern double *Hpp;
/// @brief 信息矩阵右上角
extern double *Hpm;
/// @brief 信息矩阵左下角
extern double *Hmp;
/// @brief 信息矩阵右下角
extern double *Hmm;
/// @brief 信息矩阵右下角的逆
extern double *Hmm_inv;
/// @brief tempH = Hpm * Hmm_inv
extern double *tempH;
/// @brief 优化变量当前迭代的变化量
extern double *delta_x;
/// @brief 滑动窗口大小
extern int WINDOW_SIZE;

/// @brief 主导帧相机系到当前帧相机的旋转矩阵
extern std::vector<std::vector<Eigen::Matrix3d>> R_cj_ci;
/// @brief 主导帧相机系到当前帧相机的平移矩阵
extern std::vector<std::vector<Eigen::Vector3d>> t_cj_ci;
/// @brief 世界坐标系到当前帧相机系的旋转矩阵
extern std::vector<std::vector<Eigen::Matrix3d>> R_cj_w;
/// @brief 主导帧相机系到当前帧body系的旋转矩阵
extern std::vector<std::vector<Eigen::Matrix3d>> R_bj_ci;
/// @brief 主导帧相机系到当前帧body系的平移矩阵
extern std::vector<std::vector<Eigen::Vector3d>> t_bj_ci;
/// @brief 主导帧body系到当前帧相机系的旋转矩阵
extern std::vector<std::vector<Eigen::Matrix3d>> R_cj_bi;

/// @brief 主导帧左目相机系到当前帧右目相机系的旋转矩阵
extern std::vector<std::vector<Eigen::Matrix3d>> R_rcj_lci;
/// @brief 主导帧左目相机系到当前帧右目相机系的平移矩阵
extern std::vector<std::vector<Eigen::Vector3d>> t_rcj_lci;
/// @brief 世界坐标系到当前帧右目相机系的旋转矩阵
extern std::vector<std::vector<Eigen::Matrix3d>> R_rcj_w;
/// @brief 主导帧body系到当前帧右目相机系的旋转矩阵
extern std::vector<std::vector<Eigen::Matrix3d>> R_rcj_bi;


/// @brief 逆深度ID点到节点ID的索引
extern std::unordered_map<int, int> pointToVertex_;
/// @brief 逆深度节点ID到逆深度ID的索引
extern std::unordered_map<int, int> vertexToPoint_;
/// @brief pose ID 到pose 节点ID的索引
extern std::unordered_map<int, int> poseToVertex_;
/// @brief pose 节点ID到pose ID 的索引
extern std::unordered_map<int, int> vertexToPose_;
/// @brief 外参 ID 到外参节点 ID 的索引
extern std::unordered_map<int, int> extToVertex_;
/// @brief 外参节点 ID 到外参 ID 的索引
extern std::unordered_map<int, int> vertexToExt_;

//双目相机左目到右目的变换
extern Eigen::Matrix3d R_rc_lc;
extern Eigen::Vector3d t_rc_lc;

extern std::unordered_map<int, std::pair<int, int>> startFrame_;

/// @brief 外参维度大小
extern int ext_size_;
/// @brief 位姿维度大小
extern int pose_size_;
/// @brief IMU维度大小
extern int motion_size_;
/// @brief 地图点维度大小
extern int landmarks_size_;


/// @brief 位姿维度
extern int ordering_poses_;
/// @brief 路标点维度
extern int ordering_landmarks_;
/// @brief 优化变量总维度
extern int ordering_generic_;

/// @brief 相机外参
extern std::vector<Eigen::Matrix3d> Ric;
extern std::vector<Eigen::Matrix3d> Rci;
extern std::vector<Eigen::Vector3d> tic;
extern std::vector<Eigen::Vector3d> tci;
/**
 * @brief 读取相关参数函数
 * 
 * @param config_file 文件路径
 */
void readParameters(std::string config_file);

/**
 * @brief 参数长度
 * 
 */
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};
/**
 * @brief 预积分中个参数对应序号
 * 
 */
enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

}


#endif