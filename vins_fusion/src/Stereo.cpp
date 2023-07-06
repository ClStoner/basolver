#include "core/Core.h"

using namespace core;

// msg queue
std::mutex mBuf;
std::condition_variable con;
std::queue<VisualInfo> imgBuf;
std::deque<sensor_msgs::ImuConstPtr> imuBuf;

// params
std::string configPath;
std::string leftImgTopic, rightImgTopic, depthImgTopic, l2rImgTopic, p2cImgTopic, imuTopic;

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

    leftImgTopic = params.leftImgTopic;
    if (params.Stereo)
    {
        rightImgTopic = params.rightImgTopic;
    }

    if (params.Depth)
    {
        depthImgTopic = params.depthImgTopic;
    }

    if (params.OpticalFlow)
    {
        l2rImgTopic = params.l2rOFImgTopic;
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

    // double dt = timeStamp - prevImuTimeStamp;
    // LOG_IF(INFO, dt > 0.008) << std::fixed << "at " << timeStamp << " imu dt: " << 1000 * dt;
    // prevImuTimeStamp = timeStamp;

    // imuData = std::ofstream("/root/myGit/mycode/vins_rgbd/src/vins_rgbd/test_rgbd/results/ImuData.txt", std::ofstream::app);
    // imuData << std::fixed << msg->header.stamp.toSec() << " " << msg->linear_acceleration.x << " " << msg->linear_acceleration.y << " "
    //         << msg->linear_acceleration.z << " " << msg->angular_velocity.x << " " << msg->angular_velocity.y << " " << msg->angular_velocity.z
    //         << std::endl;
}

void imgStereoCallBack(const sensor_msgs::ImageConstPtr &leftMsg, const sensor_msgs::ImageConstPtr &rightMsg)
{
    std::unique_lock<std::mutex> lock(mBuf);

    VisualInfo info;
    info.timeStamp = leftMsg->header.stamp.toSec();
    info.leftImg = utility::getMonoImgFromROS(leftMsg);
    info.rightImg = utility::getMonoImgFromROS(rightMsg);
    imgBuf.push(info);
    con.notify_one();
}

void imgDepthCallBack(const sensor_msgs::ImageConstPtr &leftMsg, const sensor_msgs::ImageConstPtr &rightMsg,
                      const sensor_msgs::ImageConstPtr &depthMsg)
{
    std::unique_lock<std::mutex> lock(mBuf);

    VisualInfo info;
    info.timeStamp = leftMsg->header.stamp.toSec();
    info.leftImg = utility::getMonoImgFromROS(leftMsg);
    info.rightImg = utility::getMonoImgFromROS(rightMsg);
    info.depthImg = utility::getDepthImgFromROS(depthMsg);
    imgBuf.push(info);
    con.notify_one();
}

void imgOFCallBack(const sensor_msgs::ImageConstPtr &leftMsg, const sensor_msgs::ImageConstPtr &rightMsg, const sensor_msgs::ImageConstPtr &p2cMsg,
                   const sensor_msgs::ImageConstPtr &l2rMsg)
{
    std::unique_lock<std::mutex> lock(mBuf);

    VisualInfo info;
    info.timeStamp = leftMsg->header.stamp.toSec();
    info.leftImg = utility::getMonoImgFromROS(leftMsg);
    info.rightImg = utility::getMonoImgFromROS(rightMsg);
    info.p2cImg = utility::getOFImgFromROS(p2cMsg);
    info.l2rImg = utility::getOFImgFromROS(l2rMsg);

    imgBuf.push(info);
    con.notify_one();
}

void imgFusionCallBack(const sensor_msgs::ImageConstPtr &leftMsg, const sensor_msgs::ImageConstPtr &rightMsg,
                       const sensor_msgs::ImageConstPtr &depthMsg, const sensor_msgs::ImageConstPtr &p2cMsg, const sensor_msgs::ImageConstPtr &l2rMsg)
{
    std::unique_lock<std::mutex> lock(mBuf);

    VisualInfo info;
    info.timeStamp = leftMsg->header.stamp.toSec();
    info.leftImg = utility::getMonoImgFromROS(leftMsg);
    info.rightImg = utility::getMonoImgFromROS(rightMsg);
    info.depthImg = utility::getDepthImgFromROS(depthMsg);
    info.p2cImg = utility::getOFImgFromROS(p2cMsg);
    info.l2rImg = utility::getOFImgFromROS(l2rMsg);
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
        if (params.Stereo && !params.Depth && !params.OpticalFlow)
        {
            std::unique_lock<std::mutex> lock(mBuf);
            con.wait(lock, [&] { return getFusionMeasurements(); });

            // TODO Stereo && Depth
            VisualInfo info = imgBuf.front();
            double timeStamp = info.timeStamp;
            imgBuf.pop();
            std::vector<std::pair<double, Eigen::Vector3d>> gyr = utility::getImuFromROS(timeStamp + params.td, imuBuf).second;
            lock.unlock();  // free lock

            slam->grabImage(timeStamp, info.leftImg, info.rightImg, gyr);
        }
        else if (params.Stereo && params.Depth && !params.OpticalFlow)
        {
            std::unique_lock<std::mutex> lock(mBuf);
            con.wait(lock, [&] { return getFusionMeasurements(); });

            // TODO Stereo && Depth
            VisualInfo info = imgBuf.front();
            double timeStamp = info.timeStamp;
            imgBuf.pop();
            std::vector<std::pair<double, Eigen::Vector3d>> gyr = utility::getImuFromROS(timeStamp + params.td, imuBuf).second;
            lock.unlock();  // free lock

            slam->grabImage(timeStamp, info.leftImg, info.rightImg, info.depthImg, gyr);
        }
        else if (params.Stereo && !params.Depth && params.OpticalFlow)
        {
            std::unique_lock<std::mutex> lock(mBuf);
            con.wait(lock, [&] { return getFusionMeasurements(); });

            // TODO Stereo && Depth
            VisualInfo info = imgBuf.front();
            double timeStamp = info.timeStamp;
            imgBuf.pop();
            std::vector<std::pair<double, Eigen::Vector3d>> gyr = utility::getImuFromROS(timeStamp + params.td, imuBuf).second;
            lock.unlock();  // free lock

            slam->grabImage(timeStamp, info.leftImg, info.rightImg, info.p2cImg, info.l2rImg, gyr);
        }
        else if (params.Stereo && params.Depth && params.OpticalFlow)
        {
            std::unique_lock<std::mutex> lock(mBuf);
            con.wait(lock, [&] { return getFusionMeasurements(); });

            // TODO Stereo && Depth
            VisualInfo info = imgBuf.front();
            double timeStamp = info.timeStamp;
            imgBuf.pop();
            std::vector<std::pair<double, Eigen::Vector3d>> gyr = utility::getImuFromROS(timeStamp + params.td, imuBuf).second;
            lock.unlock();  // free lock

            slam->grabImage(timeStamp, info.leftImg, info.rightImg, info.depthImg, info.p2cImg, info.l2rImg, gyr);
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
    ros::Subscriber subImu = n.subscribe<sensor_msgs::Imu>(imuTopic, 10000, imuCallBack, ros::TransportHints().tcpNoDelay());
    message_filters::Subscriber<sensor_msgs::Image> subLeftImg(n, leftImgTopic, 100);
    message_filters::Subscriber<sensor_msgs::Image> subRightImg(n, rightImgTopic, 100);
    message_filters::Subscriber<sensor_msgs::Image> subDepthImg(n, depthImgTopic, 100);
    message_filters::Subscriber<sensor_msgs::Image> subl2rImg(n, l2rImgTopic, 100);
    message_filters::Subscriber<sensor_msgs::Image> subp2cImg(n, p2cImgTopic, 100);

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy2Img;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> syncPolicy3Img;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> syncPolicy4Img;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
                                                      sensor_msgs::Image>
        syncPolicy5Img;

    message_filters::Synchronizer<syncPolicy2Img> sync2Img(syncPolicy2Img(10));
    message_filters::Synchronizer<syncPolicy3Img> sync3Img(syncPolicy3Img(10));
    message_filters::Synchronizer<syncPolicy4Img> sync4Img(syncPolicy4Img(10));
    message_filters::Synchronizer<syncPolicy5Img> sync5Img(syncPolicy5Img(10));

    if (params.Stereo && !params.Depth && !params.OpticalFlow)
    {
        sync2Img.connectInput(subLeftImg, subRightImg);
        sync2Img.registerCallback(boost::bind(imgStereoCallBack, _1, _2));
    }
    else if (params.Stereo && params.Depth && !params.OpticalFlow)
    {
        sync3Img.connectInput(subLeftImg, subRightImg, subDepthImg);
        sync3Img.registerCallback(boost::bind(imgDepthCallBack, _1, _2, _3));
    }
    else if (params.Stereo && !params.Depth && params.OpticalFlow)
    {
        sync4Img.connectInput(subLeftImg, subRightImg, subp2cImg, subl2rImg);
        sync4Img.registerCallback(boost::bind(imgOFCallBack, _1, _2, _3, _4));
    }
    else if (params.Stereo && params.Depth && params.OpticalFlow)
    {
        sync5Img.connectInput(subLeftImg, subRightImg, subDepthImg, subp2cImg, subl2rImg);
        sync5Img.registerCallback(boost::bind(imgFusionCallBack, _1, _2, _3, _4, _5));
    }
    else
        ROS_ERROR("Vins State error");

    std::thread syncThread(sync);

    ROS_INFO("\033[1;32m----> Vins_Fusion Started.\033[0m");

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