#include <utils/MsgTransform.h>

namespace utility
{
std::pair<std::vector<std::pair<double, Eigen::Vector3d>>, std::vector<std::pair<double, Eigen::Vector3d>>>
getImuFromROS(const double timeStamp, std::deque<sensor_msgs::ImuConstPtr> &imuBuf)
{
    std::vector<std::pair<double, Eigen::Vector3d>> acc, gyr;

    while (!imuBuf.empty())
    {
        auto imuMsg = imuBuf.front();
        if (imuMsg->header.stamp.toSec() < timeStamp)
        {
            acc.push_back(std::make_pair(imuMsg->header.stamp.toSec(), Eigen::Vector3d{imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y,
                                                                                       imuMsg->linear_acceleration.z}));
            gyr.push_back(std::make_pair(imuMsg->header.stamp.toSec(),
                                         Eigen::Vector3d{imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z}));
            imuBuf.pop_front();
        }
        else
        {
            acc.push_back(std::make_pair(imuMsg->header.stamp.toSec(), Eigen::Vector3d{imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y,
                                                                                       imuMsg->linear_acceleration.z}));
            gyr.push_back(std::make_pair(imuMsg->header.stamp.toSec(),
                                         Eigen::Vector3d{imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z}));
            break;
        }
    }

    return std::make_pair(acc, gyr);
}

cv::Mat getMonoImgFromROS(const sensor_msgs::ImageConstPtr &imgMsg)
{
    /* get mono image */
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(imgMsg, imgMsg->encoding);
    cv::Mat img;

    if (imgMsg->encoding == sensor_msgs::image_encodings::MONO8)
        cv_ptr->image.copyTo(img);
    else
        LOG(ERROR) << "input img type error...";

    return img;
}

cv::Mat getRGBImgFromROS(const sensor_msgs::ImageConstPtr &imgMsg)
{
    /* get rgb image */
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(imgMsg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img;

    if (imgMsg->encoding == sensor_msgs::image_encodings::RGB8)
        cv_ptr->image.copyTo(img);
    else
        LOG(ERROR) << "input img type error...";

    return img;
}

cv::Mat getDepthImgFromROS(const sensor_msgs::ImageConstPtr &imgMsg)
{
    /* get depth image */
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(imgMsg, imgMsg->encoding);
    cv::Mat img;

    if (imgMsg->encoding == sensor_msgs::image_encodings::TYPE_16UC1)
        cv_ptr->image.copyTo(img);
    else
        LOG(ERROR) << "input depth type error...";

    if (params.depthScale != 1 || img.type() != CV_32F)
        img.convertTo(img, CV_32F, params.depthScale);

    return img;
}

cv::Mat getOFImgFromROS(const sensor_msgs::ImageConstPtr &imgMsg)
{
    /* get depth image */
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(imgMsg, imgMsg->encoding);
    cv::Mat img;

    if (imgMsg->encoding == sensor_msgs::image_encodings::TYPE_32FC2)
        cv_ptr->image.copyTo(img);
    else
        LOG(ERROR) << "input of type error...";

    return img;
}

void Eigen2Point(const Eigen::Vector3d &v, geometry_msgs::Point &p)
{
    p.x = v.x();
    p.y = v.y();
    p.z = v.z();
}

}  // namespace utility