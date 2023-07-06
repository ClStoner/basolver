#pragma once

#include <utils/Params.h>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

namespace utility
{
std::pair<std::vector<std::pair<double, Eigen::Vector3d>>, std::vector<std::pair<double, Eigen::Vector3d>>>
getImuFromROS(const double timeStamp, std::deque<sensor_msgs::ImuConstPtr> &imuBuf);

cv::Mat getMonoImgFromROS(const sensor_msgs::ImageConstPtr &imgMsg);

cv::Mat getRGBImgFromROS(const sensor_msgs::ImageConstPtr &imgMsg);

cv::Mat getDepthImgFromROS(const sensor_msgs::ImageConstPtr &imgMsg);

cv::Mat getOFImgFromROS(const sensor_msgs::ImageConstPtr &imgMsg);

void Eigen2Point(const Eigen::Vector3d &v, geometry_msgs::Point &p);

}  // namespace utility