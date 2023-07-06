/**
 * @file Core.h
 * @author OkayManXi (786783021@qq.com)
 * @brief System类, 内部存在tracker、estimator以及visualizer等具体算法实现类
 * @version 0.1
 * @date 2022-08-11
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "core/Tracker.h"
#include "core/Estimator.h"

namespace core
{

class System
{
  public:
    POINTER_TYPEDEFS(System);
    System(ros::NodeHandle &n, const std::string &configPath)
    {
        LOG(INFO) << "System create";

        visualizer = std::make_shared<Visualization>();
        visualizer->registerPub(n);
        std::string initfile;
        n.param<std::string>("initPath", initfile, "");
        // create worker
        estimator = std::make_shared<Estimator>(visualizer);
        estimator->setConfigFile(configPath, initfile);
        estimator->setParameter();
        tracker = std::make_shared<Tracker>(estimator, visualizer);
    }
    /**
     * @brief 单目输入视觉
     *
     * @param timeStamp 输入img时间戳
     * @param img1 左目
     * @param gyr 帧间imu的角速度
     */
    // inference
    void grabImage(const double timeStamp, const cv::Mat &img1,
                   const std::vector<std::pair<double, Eigen::Vector3d>> &gyr = std::vector<std::pair<double, Eigen::Vector3d>>())
    {
        // check
        if (img1.empty())
            return;

        if (!params.Stereo)
            tracker->trackMono(timeStamp, img1, gyr);
        else
            LOG(FATAL) << "system state error ...";
    }
    /**
     * @brief 双目输入视觉/单目+深度/单目+光流
     *
     * @param timeStamp 输入img时间戳
     * @param img1 左目
     * @param img2 右目
     * @param gyr 帧间imu的角速度
     */
    // inference
    void grabImage(const double timeStamp, const cv::Mat &img1, const cv::Mat &img2,
                   const std::vector<std::pair<double, Eigen::Vector3d>> &gyr = std::vector<std::pair<double, Eigen::Vector3d>>())
    {
        // check
        if (img1.empty() || img2.empty())
            return;

        if (params.Stereo && !params.Depth && !params.OpticalFlow)
            tracker->trackStereo(timeStamp, img1, img2, gyr);
        else if (!params.Stereo && params.Depth && !params.OpticalFlow)
            tracker->trackMonoDepth(timeStamp, img1, img2, gyr);
        else if (!params.Stereo && !params.Depth && params.OpticalFlow)
            tracker->trackMonoOF(timeStamp, img1, img2, gyr);
        else
            LOG(FATAL) << "system state error ...";
    }
    /**
     * @brief 双目+深度输入视觉/单目+深度+光流
     *
     * @param timeStamp 输入img时间戳
     * @param img1 左目
     * @param img2 右目
     * @param img3 深度
     * @param gyr 帧间imu的角速度
     */
    void grabImage(const double timeStamp, const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &img3,
                   const std::vector<std::pair<double, Eigen::Vector3d>> &gyr = std::vector<std::pair<double, Eigen::Vector3d>>())
    {
        // check
        if (img1.empty() || img2.empty() || img3.empty())
            return;

        if (params.Stereo && params.Depth && !params.OpticalFlow)
            tracker->trackStereoDepth(timeStamp, img1, img2, img3, gyr);
        else if (!params.Stereo && params.Depth && params.OpticalFlow)
            tracker->trackMonoFusion(timeStamp, img1, img2, img3, gyr);
        else
            LOG(FATAL) << "system state error ...";
    }
    /**
     * @brief 双目+光流输入视觉
     *
     * @param timeStamp 输入img时间戳
     * @param img1 左目
     * @param img2 右目
     * @param img3 前后帧光流
     * @param img4 左右目光流
     * @param gyr 帧间imu的角速度
     */
    void grabImage(const double timeStamp, const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &img3, const cv::Mat &img4,
                   const std::vector<std::pair<double, Eigen::Vector3d>> &gyr = std::vector<std::pair<double, Eigen::Vector3d>>())
    {
        // check
        if (img1.empty() || img2.empty() || img3.empty() || img4.empty())
            return;

        if (params.OpticalFlow)
        {
            cv::Mat leftImg = img1;
            cv::Mat rightImg = img2;
            cv::Mat p2cImg = img3;
            cv::Mat l2rImg = img4;

            tracker->trackStereoOF(timeStamp, leftImg, rightImg, p2cImg, l2rImg, gyr);
        }

        else
            LOG(FATAL) << "system state error ...";
    }
    /**
     * @brief 双目+深度+光流输入视觉
     *
     * @param timeStamp 输入img时间戳
     * @param img1 左目
     * @param img2 右目
     * @param img3 深度
     * @param img4 前后帧光流
     * @param img5 左右目光流
     * @param gyr 帧间imu的角速度
     */
    void grabImage(const double timeStamp, const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &img3, const cv::Mat &img4, const cv::Mat &img5,
                   const std::vector<std::pair<double, Eigen::Vector3d>> &gyr = std::vector<std::pair<double, Eigen::Vector3d>>())
    {
        // check
        if (img1.empty() || img2.empty() || img3.empty() || img4.empty() || img5.empty())
            return;

        if (params.Depth && params.OpticalFlow)
        {
            cv::Mat leftImg = img1;
            cv::Mat rightImg = img2;
            cv::Mat depthImg = img3;
            cv::Mat p2cImg = img4;
            cv::Mat l2rImg = img5;

            tracker->trackStereoFusion(timeStamp, leftImg, rightImg, depthImg, p2cImg, l2rImg, gyr);
        }

        else
            LOG(FATAL) << "system state error ...";
    }
    /**
     * @brief imu输入
     *
     * @param timeStamp imu时间戳
     * @param acc imu加速度
     * @param gyr imu角速度
     */
    void grabImu(const double timeStamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        estimator->inputIMU(timeStamp, acc, gyr);
    }

  private:
    // worker
    /// Visualization worker
    Visualization::Ptr visualizer;
    /// Tracker worker
    Tracker::Ptr tracker;
    /// Estimator worker
    Estimator::Ptr estimator;
};
}  // namespace core