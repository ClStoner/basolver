/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include "utils/Marco.h"
#include "utils/Utility.h"
#include "utils/Params.h"
#include "core/Frame.h"
#include "factor/IMUFactor.h"

#include <Eigen/Dense>

#include <map>
#include "../../../basolver/basolver.h"

class ImageFrame
{
  public:
    ImageFrame(){};

    ImageFrame(double timeStamp, const core::Frame &feature) : timeStamp(timeStamp), frameID(feature.frameID), is_key_frame(false)
    {
        for (auto &point : feature.points)
            points.emplace(point.id, point);
    };

    // frame info
    double timeStamp;
    long frameID;
    bool is_key_frame;

    // pose
    Eigen::Matrix3d R;
    Eigen::Vector3d T;

    // measurement
    // IntegrationBase::Ptr pre_integration;
    BaSolver::IntegrationBase *pre_integration;
    std::map<long, PointInfo> points;
};

class InitialAlignment
{
  public:
    InitialAlignment() : windowSize(params.windowSize), gNorm(params.gNorm), tic(params.TicLeft.block(0, 3, 3, 1)) {}

    void solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs);

    bool visualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x);

    void refineGravity(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x);

    bool linearAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x);

    /**
     * @brief 双目视觉与imu对齐初始化
     * @refer VINS-Mono/Fusion initialAlignment
     * @param all_image_frame 输入视觉帧
     * @param Bgs 陀螺仪bias
     * @param g 重力方向
     * @param x 尺度
     * @return true vi对齐成功
     * @return false 失败
     */
    bool stereoVisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x);
    /**
     * @brief 双目修正重力方向
     * @refer VINS-Mono/Fusion initialAlignment
     * @param all_image_frame 输入视觉帧
     * @param g 重力方向
     * @param x 尺度
     */
    void stereoRefineGravity(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x);
    /**
     * @brief 双目vi对齐
     * @refer VINS-Mono/Fusion initialAlignment
     * @param all_image_frame 输入视觉帧
     * @param g 重力方向
     * @param x 尺度
     * @return true
     * @return false
     */
    bool stereoLinearAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x);

    // params
    const int windowSize;
    const double gNorm;
    const Eigen::Vector3d tic;
};
