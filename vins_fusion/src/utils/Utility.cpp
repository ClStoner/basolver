#include <utils/Utility.h>

namespace utility
{

cv::Mat getDepthImgFromStereo(const cv::Mat &leftImg, const cv::Mat &rightImg)
{
    cv::Mat dispImg(leftImg.rows, leftImg.cols, CV_16SC1), depthImg(leftImg.rows, leftImg.cols, CV_16UC1);
    if (leftImg.empty() || rightImg.empty())
        return depthImg;

    // static cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm = cv::stereo::StereoBinarySGBM::create(0, 32, 3);
    static cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 64, 4);
    // sgbm->setP1(8 * leftImg.channels() * 4 * 4);
    // sgbm->setP2(8 * leftImg.channels() * 4 * 4);
    // sgbm->setDisp12MaxDiff(-1);
    // sgbm->setPreFilterCap(1);
    // sgbm->setUniquenessRatio(10);
    // sgbm->setSpeckleWindowSize(100);
    // sgbm->setSpeckleRange(100);
    sgbm->setMode(cv::StereoSGBM::MODE_HH);

    sgbm->compute(leftImg, rightImg, dispImg);

    double bf = 0.05 * 423.00726318359375;
    for (int i = 0; i < dispImg.rows * dispImg.cols; i++)
    {
        double disparity = (double)dispImg.at<short>(i) / 16.0;
        if (dispImg.at<short>(i) == 0)
            depthImg.at<uint16_t>(i) = 0;
        else
            // depthImg.at<uchar>(i) = static_cast<uchar>(bf / disparity);
            depthImg.at<uint16_t>(i) = static_cast<uint16_t>(bf / disparity);
    }

    return depthImg;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr getPointCloudFromMat(const cv::Mat &depthImage, const cv::Mat &K, const double depthTh)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>());
    const int rows = depthImage.rows;
    const int cols = depthImage.cols;

    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            pcl::PointXYZ point;
            float depth = depthImage.ptr<float>(v)[u];

            if (depth > depthTh)
                continue;

            // transform to enu coordinate
            // point.x = depth;
            // point.y = -(u - K.at<double>(0, 2)) * depth / K.at<double>(0, 0);
            // point.z = -(v - K.at<double>(1, 2)) * depth / K.at<double>(1, 1);
            // transform to camera coordinate
            point.x = (u - K.at<double>(0, 2)) * depth / K.at<double>(0, 0);
            point.y = (v - K.at<double>(1, 2)) * depth / K.at<double>(1, 1);
            point.z = depth;

            pointCloud->points.push_back(point);
        }
    }

    pointCloud->height = 1;
    pointCloud->width = pointCloud->points.size();
    pointCloud->is_dense = true;

    return pointCloud;
}

Eigen::Matrix3d getRotationByImu(const double dt, const Eigen::Matrix3d &Ric, const std::vector<std::pair<double, Eigen::Vector3d>> &gyr)
{
    Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
    if (gyr.size() < 4)
        return rot;

    // predict next pose. Assume constant velocity motion
    Eigen::Vector3d meanAngVel{0, 0, 0};
    for (auto &iter : gyr)
        meanAngVel += iter.second;
    meanAngVel *= 1.0f / gyr.size();
    meanAngVel = Ric.transpose() * meanAngVel;

    // rotation between curr camera to prev camera
    rot = Eigen::AngleAxisd{meanAngVel.norm() * dt, meanAngVel.normalized()}.toRotationMatrix();
    return rot.transpose();
}

}  // namespace utility