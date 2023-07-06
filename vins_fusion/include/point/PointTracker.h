/**
 * @file PointTracker.h
 * @author OkayManXi (786783021@qq.com)
 * @brief 特帧点追踪算法实现
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
#include "core/Frame.h"
#include "FeatureDetector.h"

#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/CataCamera.h>
#include <camodocal/camera_models/PinholeCamera.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/optflow.hpp>

using namespace core;
/**
 * @brief PointTracker类实现了特帧点追踪以及提取
 * 参考了部分VINS-Mono，Vins-Fusion的前端算法
 */
class PointTracker
{

  public:
    POINTER_TYPEDEFS(PointTracker);
    PointTracker();

    /**
     * @brief 特帧追踪接口
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param prev 参考帧
     * @param curr 当前帧
     * @param isKeyframe 是否关键帧
     */
    void inference(const Frame::Ptr &prev, const Frame::Ptr &curr, const bool isKeyframe = false);
    /**
     * @brief imu预测角点位置
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param frame 输入帧
     * @param R imu预测姿态变换
     */
    void setPredictPoints(const Frame::Ptr &frame, const Eigen::Matrix3d &R);
    /**
     * @brief 可视化角点
     *
     * @param frame 可视化帧
     * @param isShow 是否ishow
     * @return cv::Mat 可视化img
     */
    cv::Mat drawPoints(const Frame::Ptr &frame, const bool isShow = false);

  private:
    /**
     * @brief 依据追踪周期排序
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param frame 输入帧
     * @return cv::Mat Mask
     */
    cv::Mat sortPointsByCnt(const Frame::Ptr &frame);
    /**
     * @brief 提取角点
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param prev 参考帧
     * @param curr 当前帧
     */
    void extractPoints(const Frame::Ptr &prev, const Frame::Ptr &curr);
    /**
     * @brief 基础矩阵去除外点
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param prev 参考帧
     * @param curr 当前帧
     */
    void rejectWithF(const Frame::Ptr &prev, const Frame::Ptr &curr);

    /**
     * @brief 帧间LK光流
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param prev 参考帧
     * @param curr 当前帧
     */
    void matchByLKOpticFlow(const Frame::Ptr &prev, const Frame::Ptr &curr);

    /**
     * @brief 帧间硬件光流
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param prev 参考帧
     * @param curr 当前帧
     */
    void matchByFBOpticFlow(const Frame::Ptr &prev, const Frame::Ptr &curr);

    /**
     * @brief 双目LK光流
     *
     * @param frame 输入帧
     */
    void matchRightPointsByLKOpticFlow(const Frame::Ptr &frame);

    /**
     * @brief 双目硬件光流
     * @refer VINS-Mono/Fusion featureTracker and ORB-SLAM Tracker
     * @param frame 输入帧
     */
    void matchRightPointsByFBOpticFlow(const Frame::Ptr &frame);

    /**
     * @brief 点坐标是否在图像内
     *
     * @tparam T
     * @param pt 输入点2D坐标
     * @return true 在图像内
     * @return false 不在
     */
    template <typename T>
    bool inBorder(const T &pt)
    {
        int imgX = std::round(pt.x);
        int imgY = std::round(pt.y);

        return borderSize <= imgX && imgX < imgWidth - borderSize && borderSize <= imgY && imgY < imgHeight - borderSize;
    }

  private:
    // base params
    /// PointTracker cnt
    int cnt;
    /// PointTracker log freq
    int logFreq;
    /// PointTracker is stereo state
    const bool Stereo;

    // img params
    /// PointTracker img height
    const int imgHeight;
    /// PointTracker img width
    const int imgWidth;
    /// PointTracker focal length
    const double focalLength;
    /// PointTracker img border size
    const int borderSize;

    // extract
    /// PointTracker point dectector type
    const int pointExtractType;
    /// PointTracker point detect max num
    const int pointExtractMaxNum;
    /// PointTracker point detect min dist
    const int pointExtractMinDist;

    // match
    /// PointTracker ransac pixel thereshold
    const double pointMatchRansacTh;
    /// PointTracker optical flow by LK pixel thereshold
    const double precision = 0.01;

    // imu KLT
    /// PointTracker Rot from cam to imu
    const Eigen::Matrix3d Ric;
    /// PointTracker has predict point flag
    bool hasPredict;
    /// PointTracker predict points
    std::vector<cv::Point2f> predictPoints;

    // worker
    /// PointTracker camera model for left cam
    camodocal::CameraPtr camera;
    /// PointTracker fast feature detector
    FastDetector::Ptr fast;
    /// PointTracker harris feature detector
    HarrisDetector::Ptr harris;
};
