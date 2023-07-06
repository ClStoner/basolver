#pragma once

#include <opencv2/calib3d.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "utils/Marco.h"
#include "utils/TicToc.h"
#include "utils/Math.h"

namespace utility
{

cv::Mat getDepthImgFromStereo(const cv::Mat &leftImg, const cv::Mat &rightImg);

pcl::PointCloud<pcl::PointXYZ>::Ptr getPointCloudFromMat(const cv::Mat &depthImage, const cv::Mat &K, const double depth);

Eigen::Matrix3d getRotationByImu(const double dt, const Eigen::Matrix3d &Ric, const std::vector<std::pair<double, Eigen::Vector3d>> &gyr);

inline bool rotIsValid(const Eigen::Matrix3d &R)
{
    if (std::abs(R.determinant()) < 0.9)
        return false;

    return true;
}

// tmeplate
template <typename T> void reduceVector(std::vector<T> &inputVector, const std::vector<uchar> &status)
{
    int j = 0;
    for (unsigned i = 0; i < inputVector.size(); i++)
        if (status[i])
            inputVector[j++] = inputVector[i];
    inputVector.resize(j);
}

}  // namespace utility
