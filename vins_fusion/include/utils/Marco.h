/**
 * @file Marco.h
 * @author OkayManXi (786783021@qq.com)
 * @brief 宏定义以及部分数据结构和枚举
 * @version 0.1
 * @date 2022-08-11
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <fstream>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#include <queue>

/**
 * @brief 特征点提取类别
 *
 */
enum PointExtractFlag
{
    FAST = 0,
    Harris = 1,
    None = 2
};

/**
 * @brief 图像可视化类别
 *
 */
enum VisualizationFlag
{
    Odometry = 0,
    PointExtract = 1,
    PointClouds = 6,
    SpatialPoint = 7,
    MarginPoint = 8,
    HFNetPoint = 11,
    LoopMatch = 12

};
/**
 * @brief ceres 优化参数自由度
 *
 */
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_POINT = 1
};
/**
 * @brief 状态矩阵中状态量位置
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
/**
 * @brief 噪声矩阵中bg ba4个噪声位置
 *
 */
enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
/**
 * @brief ROS接口中视觉输入同步数据结构
 *
 */
struct VisualInfo
{
    /// VisualInfo timeStamp
    double timeStamp;
    /// VisualInfo input left img
    cv::Mat leftImg;
    /// VisualInfo input right img
    cv::Mat rightImg;
    /// VisualInfo input depth img
    cv::Mat depthImg;
    /// VisualInfo input prev2curr opticalflow img
    cv::Mat p2cImg;
    /// VisualInfo input left2right opticalflow img
    cv::Mat l2rImg;
};
/**
 * @brief 视觉前端中特帧点的数据结构
 *
 */
struct PointInfo
{
    /// Point featureid
    long id;
    /// Point track cnt
    int cnt;

    // Mono info
    /// Point location uv
    Eigen::Vector2d p2D;  // uv
    /// Point norm location
    Eigen::Vector3d p3D;  // norm
    /// Point location vel
    Eigen::Vector2d velocity;

    // Base Point info
    /// Point depth
    double depth;

    // Stereo info
    /// Point has stereo obs flag
    bool stereo;
    /// Point location uv right
    Eigen::Vector2d p2DRight;
    /// Point norm location right
    Eigen::Vector3d p3DRight;
    /// Point location vel
    Eigen::Vector2d velocityRight;

    PointInfo() : depth(0), stereo(false) {}
};
/**
 * @brief 关键帧数据结构
 *
 */
struct Keyframe
{
    /// Keyframe timestamp
    double timeStamp;
    /// Keyframe Ric extrinsic
    Eigen::Matrix3d Ric;
    /// Keyframe tic extrinsic
    Eigen::Vector3d tic;
    /// Keyframe R from body to world
    Eigen::Matrix3d R;
    /// Keyframe t from body to world
    Eigen::Vector3d t;
    /// Keyframe  point 3d loc
    std::vector<Eigen::Vector3d> pts3D;
    /// Keyframe  point 2d norm loc
    std::vector<Eigen::Vector3d> ptsUn;
    /// Keyframe  point 2d uv loc
    std::vector<Eigen::Vector2d> ptsUv;
    /// Keyframe  point feature id
    std::vector<long> ptsID;
};
/**
 * @brief 视觉前端可视化数据结构
 *
 */
struct VisualizationImg
{
    /// visualization timestamp
    double timeStamp;
    /// visualization frame id
    long frameID;
    /// visualization depth img to visual pointcloud
    cv::Mat depthImg;
    /// visualization point track img
    cv::Mat pointImg;
};
/**
 * @brief 滑窗地图点可视化数据结构
 *
 */
struct Visualization3D
{
    /// Visualization3D timstamp
    double timeStamp;
    /// Visualization3D frame id
    long frameID;
    /// Visualization3D point 3d loc
    std::vector<Eigen::Vector3d> pts3D;
    /// Visualization3D margin point 3d loc
    std::vector<Eigen::Vector3d> marginPts3D;
};

/**
 * @brief 滑窗位姿可视化数据结构
 *
 */
struct VisualizationOdom
{
    /// VisualizationOdom timetamp
    double timeStamp;
    /// VisualizationOdom frame id
    long frameID;
    /// VisualizationOdom is keyframe flag
    bool keyframe;
    /// VisualizationOdom t from body to world
    Eigen::Vector3d P;
    /// VisualizationOdom v from body to world
    Eigen::Vector3d V;
    /// VisualizationOdom Rot from body to world
    Eigen::Quaterniond R;
    /// VisualizationOdom Rot from cam to imu
    std::vector<Eigen::Quaterniond> Ric;
    /// VisualizationOdom t from cam to imu
    std::vector<Eigen::Vector3d> tic;

    VisualizationOdom() : keyframe(false) {}
};

/**
 * @brief 指针宏
 *
 */
#define POINTER_TYPEDEFS(TypeName)                                                                                                                   \
    typedef std::shared_ptr<TypeName> Ptr;                                                                                                           \
    typedef std::shared_ptr<const TypeName> ConstPtr;                                                                                                \
    typedef std::unique_ptr<TypeName> UniquePtr;                                                                                                     \
    typedef std::weak_ptr<TypeName> WeakPrt;                                                                                                         \
    typedef std::weak_ptr<const TypeName> WeakConstPtr;                                                                                              \
    void definePointerTypedefs##__FILE__##__LINE__(void)
