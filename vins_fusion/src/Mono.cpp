#include "core/Core.h"

using namespace core;

// msg queue
std::mutex mBuf;
std::condition_variable con;
std::queue<VisualInfo> imgBuf;
std::deque<sensor_msgs::ImuConstPtr> imuBuf;

// params
std::string configPath;
std::string leftImgTopic, depthImgTopic, p2cImgTopic, imuTopic;

// worker
ParamManager::Ptr options;
System::Ptr slam;

void initParams(ros::NodeHandle &n)
{
    n.param<std::string>("configPath", configPath, "");
    ROS_INFO("configPath : %s", configPath.c_str());

    // init worker
    options = std::make_shared<ParamManager>(configPath);
    slam = std::make_shared<System>(n, configPath);

    if (!params.Stereo)
    {
        leftImgTopic = params.leftImgTopic;
    }

    if (params.Depth)
    {
        depthImgTopic = params.depthImgTopic;
    }

    if (params.OpticalFlow)
    {
        p2cImgTopic = params.p2cOFImgTOpic;
    }

    imuTopic = params.imuTopic;
}

bool imuAvailable(const double t)
{
    if (!imuBuf.empty() && t <= imuBuf.back()->header.stamp.toSec())
        return true;
    else
        return false;
}

void imuCallBack(const sensor_msgs::ImuConstPtr &msg)
{
    std::unique_lock<std::mutex> lock(mBuf);
    imuBuf.push_back(msg);

    double timeStamp = msg->header.stamp.toSec();
    Eigen::Vector3d acc{msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z};
    Eigen::Vector3d gyr{msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z};
    slam->grabImu(timeStamp, acc, gyr);
    con.notify_one();
}

void imgMonoCallBack(const sensor_msgs::ImageConstPtr &leftMsg)
{
    std::unique_lock<std::mutex> lock(mBuf);

    VisualInfo info;
    info.timeStamp = leftMsg->header.stamp.toSec();
    info.leftImg = utility::getMonoImgFromROS(leftMsg);
    imgBuf.push(info);
    con.notify_one();
}

void imgDepthOrOFCallBack(const sensor_msgs::ImageConstPtr &leftMsg, const sensor_msgs::ImageConstPtr &depthOFMsg)
{
    std::unique_lock<std::mutex> lock(mBuf);

    VisualInfo info;
    info.timeStamp = leftMsg->header.stamp.toSec();
    info.leftImg = utility::getMonoImgFromROS(leftMsg);
    if (params.Depth)
        info.depthImg = utility::getDepthImgFromROS(depthOFMsg);
    else if (params.OpticalFlow)
        info.p2cImg = utility::getOFImgFromROS(depthOFMsg);

    imgBuf.push(info);
    con.notify_one();
}

void imgFusionCallBack(const sensor_msgs::ImageConstPtr &leftMsg, const sensor_msgs::ImageConstPtr &depthMsg,
                       const sensor_msgs::ImageConstPtr &p2cMsg)
{
    std::unique_lock<std::mutex> lock(mBuf);

    VisualInfo info;
    info.timeStamp = leftMsg->header.stamp.toSec();
    info.leftImg = utility::getMonoImgFromROS(leftMsg);
    info.depthImg = utility::getDepthImgFromROS(depthMsg);
    info.p2cImg = utility::getOFImgFromROS(p2cMsg);
    imgBuf.push(info);
    con.notify_one();
}

bool getFusionMeasurements()
{
    if (!imgBuf.empty() && !imuBuf.empty())
    {
        double timeStamp = imgBuf.front().timeStamp;

        return imuAvailable(timeStamp + params.td);
    }
    else
        return false;
}

void sync()
{
    while (true)
    {
        if (!params.Stereo && !params.Depth && !params.OpticalFlow)
        {
            std::unique_lock<std::mutex> lock(mBuf);
            con.wait(lock, [&] { return getFusionMeasurements(); });
            // TODO Stereo && Depth
            VisualInfo info = imgBuf.front();
            double timeStamp = info.timeStamp;
            imgBuf.pop();
            std::vector<std::pair<double, Eigen::Vector3d>> gyr = utility::getImuFromROS(timeStamp + params.td, imuBuf).second;
            lock.unlock();  // free lock

            slam->grabImage(timeStamp, info.leftImg, gyr);
        }
        else if (!params.Stereo && params.Depth && !params.OpticalFlow)
        {
            std::unique_lock<std::mutex> lock(mBuf);
            con.wait(lock, [&] { return getFusionMeasurements(); });

            // TODO Stereo && Depth
            VisualInfo info = imgBuf.front();
            double timeStamp = info.timeStamp;
            imgBuf.pop();
            std::vector<std::pair<double, Eigen::Vector3d>> gyr = utility::getImuFromROS(timeStamp + params.td, imuBuf).second;
            lock.unlock();  // free lock

            slam->grabImage(timeStamp, info.leftImg, info.depthImg, gyr);
        }
        else if (!params.Stereo && !params.Depth && params.OpticalFlow)
        {
            std::unique_lock<std::mutex> lock(mBuf);
            con.wait(lock, [&] { return getFusionMeasurements(); });

            // TODO Stereo && Depth
            VisualInfo info = imgBuf.front();
            double timeStamp = info.timeStamp;
            imgBuf.pop();
            std::vector<std::pair<double, Eigen::Vector3d>> gyr = utility::getImuFromROS(timeStamp + params.td, imuBuf).second;
            lock.unlock();  // free lock

            slam->grabImage(timeStamp, info.leftImg, info.p2cImg, gyr);
        }
        else if (!params.Stereo && params.Depth && params.OpticalFlow)
        {
            std::unique_lock<std::mutex> lock(mBuf);
            con.wait(lock, [&] { return getFusionMeasurements(); });

            // TODO Stereo && Depth
            VisualInfo info = imgBuf.front();
            double timeStamp = info.timeStamp;
            imgBuf.pop();
            std::vector<std::pair<double, Eigen::Vector3d>> gyr = utility::getImuFromROS(timeStamp + params.td, imuBuf).second;
            lock.unlock();  // free lock

            slam->grabImage(timeStamp, info.leftImg, info.depthImg, info.p2cImg, gyr);
        }
        else
            ROS_ERROR("Vins State error");
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_rgbd");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    // init config path, vocabulary path and ros subscriber topic and init manhattanslam
    initParams(n);

    // sync process by ros messageer filter: ros nodehandle ros topic queuesize
    ros::Subscriber subImg;
    ros::Subscriber subImu = n.subscribe<sensor_msgs::Imu>(imuTopic, 10000, imuCallBack, ros::TransportHints().tcpNoDelay());
    message_filters::Subscriber<sensor_msgs::Image> subLeftImg(n, leftImgTopic, 100);
    message_filters::Subscriber<sensor_msgs::Image> subDepthImg(n, depthImgTopic, 100);
    message_filters::Subscriber<sensor_msgs::Image> subp2cImg(n, p2cImgTopic, 100);

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy2Img;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> syncPolicy3Img;

    message_filters::Synchronizer<syncPolicy2Img> sync2Img(syncPolicy2Img(10));
    message_filters::Synchronizer<syncPolicy3Img> sync3Img(syncPolicy3Img(10));

    if (!params.Stereo && !params.Depth && !params.OpticalFlow)
    {
        subImg = n.subscribe(leftImgTopic, 2000, imgMonoCallBack);
    }
    else if (!params.Stereo && params.Depth && !params.OpticalFlow)
    {
        sync2Img.connectInput(subLeftImg, subDepthImg);
        sync2Img.registerCallback(boost::bind(imgDepthOrOFCallBack, _1, _2));
    }
    else if (!params.Stereo && !params.Depth && params.OpticalFlow)
    {
        sync2Img.connectInput(subLeftImg, subp2cImg);
        sync2Img.registerCallback(boost::bind(imgDepthOrOFCallBack, _1, _2));
    }
    else if (!params.Stereo && params.Depth && params.OpticalFlow)
    {
        sync3Img.connectInput(subLeftImg, subDepthImg, subp2cImg);
        sync3Img.registerCallback(boost::bind(imgFusionCallBack, _1, _2, _3));
    }
    else
        ROS_ERROR("Vins State error");

    std::thread syncThread(sync);

    ROS_INFO("\033[1;32m----> Vins_Fusion Mono Started.\033[0m");

    ros::Rate rate(500);
    bool status = ros::ok();
    while (status)
    {
        ros::spinOnce();
        status = ros::ok();
        rate.sleep();
    }

    return 0;
}