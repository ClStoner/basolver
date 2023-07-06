/**
 * @file Tracker.h
 * @author OkayManXi (786783021@qq.com)
 * @brief 视觉前端算法实现类
 * @version 0.1
 * @date 2022-08-11
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "utils/Marco.h"
#include "utils/Utility.h"
#include "utils/Params.h"
#include <utils/Visualization.h>
#include <point/PointTracker.h>
#include "core/Frame.h"
#include "core/Estimator.h"

#include <thread>

namespace core
{
/**
 * @brief Tracker类，视觉前端算法实现
 *
 */
class Tracker
{
  public:
    enum TrackerState
    {
        NOT_INITIALIZED = 0,
        OK = 1,
    };

    POINTER_TYPEDEFS(Tracker);
    Tracker(const Estimator::Ptr &estimator, const Visualization::Ptr &visualizer);

    /**
     * @brief Stereo Track
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param timeStamp 时间戳
     * @param leftImg 左目
     * @param rightImg 右目
     * @param gyr 角速度
     */
    void trackStereo(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &rightImg,
                     const std::vector<std::pair<double, Eigen::Vector3d>> &gyr);
    /**
     * @brief Stereo and Depth Track
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param timeStamp 时间戳
     * @param leftImg 左目
     * @param rightImg 右目
     * @param depthImg 深度
     * @param gyr 角速度
     */
    void trackStereoDepth(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &rightImg, const cv::Mat &depthImg,
                          const std::vector<std::pair<double, Eigen::Vector3d>> &gyr = std::vector<std::pair<double, Eigen::Vector3d>>());
    /**
     * @brief Stereo and OpticalFlow Track
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param timeStamp 时间戳
     * @param leftImg 左目
     * @param rightImg 右目
     * @param p2cImg 前后光流
     * @param l2rImg 左右光流
     * @param gyr 角速度
     */
    void trackStereoOF(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &rightImg, const cv::Mat &p2cImg, const cv::Mat &l2rImg,
                       const std::vector<std::pair<double, Eigen::Vector3d>> &gyr = std::vector<std::pair<double, Eigen::Vector3d>>());
    /**
     * @brief Stereo OpticalFlow and Depth Track
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param timeStamp 时间戳
     * @param leftImg 左目
     * @param rightImg 右目
     * @param depthImg 深度
     * @param p2cImg 前后光流
     * @param l2rImg 左右光流
     * @param gyr 角速度
     */
    void trackStereoFusion(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &rightImg, const cv::Mat &depthImg, const cv::Mat &p2cImg,
                           const cv::Mat &l2rImg, const std::vector<std::pair<double, Eigen::Vector3d>> &gyr);

    /**
     * @brief Mono Track
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param timeStamp 时间戳
     * @param leftImg 左目
     * @param gyr 角速度
     */
    void trackMono(const double timeStamp, const cv::Mat &leftImg, const std::vector<std::pair<double, Eigen::Vector3d>> &gyr);
    /**
     * @brief Mono and Depth Track
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param timeStamp 时间戳
     * @param leftImg 左目
     * @param depthImg 深度
     * @param gyr 角速度
     */
    void trackMonoDepth(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &depthImg,
                        const std::vector<std::pair<double, Eigen::Vector3d>> &gyr);
    /**
     * @brief Mono and OpticalFlow Track
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param timeStamp 时间戳
     * @param leftImg 左目
     * @param p2cImg 前后光流
     * @param gyr 角速度
     */
    void trackMonoOF(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &p2cImg,
                     const std::vector<std::pair<double, Eigen::Vector3d>> &gyr);
    /**
     * @brief Mono ,OpticalFlow and Depth Track
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param timeStamp 时间戳
     * @param leftImg 左目
     * @param depthImg 深度
     * @param p2cImg 前后光流
     * @param gyr 角速度
     */
    void trackMonoFusion(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &depthImg, const cv::Mat &p2cImg,
                         const std::vector<std::pair<double, Eigen::Vector3d>> &gyr);

  private:
    // params
    /// Tracker mutex
    std::mutex mTrack;
    /// Tracker frame counter
    long frameCnt;
    const int logFreq;

    // internal worker
    /// Tracker PointTracker worker
    PointTracker::Ptr pointTracker;

    // external worker
    /// Tracker Estimator worker
    Estimator::Ptr estimator;
    /// Tracker Visulizationn worker
    Visualization::Ptr visualizer;

    // extrinsics from camera to imu
    /// Tracker left cam to imu extrinsic
    Eigen::Matrix4d TicLeft;
    /// Tracker right cam to imu extrinsic
    Eigen::Matrix4d TicRight;

    // state
    /// Tracker refer, prev, curr frame
    Frame::Ptr refer, prev, curr;
};
}  // namespace core
