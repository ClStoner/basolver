#include "core/Tracker.h"

namespace core
{
Tracker::Tracker(const Estimator::Ptr &estimator, const Visualization::Ptr &visualizer)
    : frameCnt(-1), logFreq(params.logFreq), TicLeft(params.TicLeft), TicRight(params.TicRight)
{
    LOG(INFO) << "Tracker create";

    // init internal worker
    pointTracker = std::make_shared<PointTracker>();

    // init external worker
    this->estimator = estimator;
    this->visualizer = visualizer;

    // init state
    Frame::frameIDCnt = 0;
    Frame::pointIDCnt = 0;
    Frame::leftCam = params.leftCam;
    if (params.clahe)
        Frame::clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    if (params.Stereo)
        Frame::rightCam = params.rightCam;
}

void Tracker::trackStereo(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &rightImg,
                          const std::vector<std::pair<double, Eigen::Vector3d>> &gyr)
{
    std::unique_lock<std::mutex> lock(mTrack);

    utility::TicToc td;
    frameCnt++;

    curr = std::make_shared<Frame>(timeStamp, leftImg, rightImg);

    // set lk optic flow predict by gyr
    if (!gyr.empty() && prev && !prev->points.empty())
    {
        double dt = curr->timeStamp - prev->timeStamp;
        Eigen::Matrix3d Ric = TicLeft.block(0, 0, 3, 3);
        Eigen::Matrix3d rot = utility::getRotationByImu(dt, Ric, gyr);  // prev to curr
        pointTracker->setPredictPoints(prev, rot);
    }

    // control frame freq
    // klt track points
    if (frameCnt % params.freqRatio != 0)
        pointTracker->inference(prev, curr, false);
    else
    {
        // extract points, lines and planes only when publish frame
        pointTracker->inference(prev, curr, true);

        estimator->inputFeature(*curr);

        // update refer frame
        refer = curr;
    }

    // update prev frame
    prev = curr;

    // visualization
    VisualizationImg visualImg;
    visualImg.timeStamp = curr->timeStamp;
    visualImg.frameID = curr->frameID;
    visualImg.depthImg = curr->depthImg;
    visualImg.pointImg = pointTracker->drawPoints(curr, false);
    visualizer->inputImg(visualImg);

    LOG_EVERY_N(INFO, logFreq) << "Tracker inference cost: " << td.toc() << "ms";
}

void Tracker::trackStereoDepth(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &rightImg, const cv::Mat &depthImg,
                               const std::vector<std::pair<double, Eigen::Vector3d>> &gyr)
{
    std::unique_lock<std::mutex> lock(mTrack);

    utility::TicToc td;
    frameCnt++;

    curr = std::make_shared<Frame>(timeStamp, leftImg, rightImg, depthImg);

    // set lk optic flow predict by gyr
    if (!gyr.empty() && prev && !prev->points.empty())
    {
        double dt = curr->timeStamp - prev->timeStamp;
        Eigen::Matrix3d Ric = TicLeft.block<3, 3>(0, 0);
        Eigen::Matrix3d rot = utility::getRotationByImu(dt, Ric, gyr);  // prev to curr
        pointTracker->setPredictPoints(prev, rot);
    }

    // control frame freq
    // klt track points
    if (frameCnt % params.freqRatio != 0)
        pointTracker->inference(prev, curr, false);
    else
    {
        // extract points, lines and planes only when publish frame
        pointTracker->inference(prev, curr, true);

        // visualization
        VisualizationImg visual;
        visual.timeStamp = timeStamp;
        visual.frameID = curr->frameID;
        visual.pointImg = pointTracker->drawPoints(curr, false);

        estimator->inputFeature(*curr);

        // update refer frame
        refer = curr;
    }

    // update prev frame
    prev = curr;

    LOG_EVERY_N(INFO, logFreq) << "Tracker inference cost: " << td.toc() << "ms";
}

void Tracker::trackStereoOF(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &rightImg, const cv::Mat &p2cImg, const cv::Mat &l2rImg,
                            const std::vector<std::pair<double, Eigen::Vector3d>> &gyr)
{
    std::unique_lock<std::mutex> lock(mTrack);

    utility::TicToc td;
    frameCnt++;

    curr = std::make_shared<Frame>(timeStamp, leftImg, rightImg, p2cImg, l2rImg);

    // set lk optic flow predict by gyr
    if (!gyr.empty() && prev && !prev->points.empty())
    {
        double dt = curr->timeStamp - prev->timeStamp;
        Eigen::Matrix3d Ric = TicLeft.block<3, 3>(0, 0);
        Eigen::Matrix3d rot = utility::getRotationByImu(dt, Ric, gyr);  // prev to curr
        pointTracker->setPredictPoints(prev, rot);
    }

    // control frame freq
    // klt track points
    if (frameCnt % params.freqRatio != 0)
        pointTracker->inference(prev, curr, false);
    else
    {
        // extract points, lines and planes only when publish frame
        pointTracker->inference(prev, curr, true);

#ifdef TEST_PERF
        {
            static double sum_dt = 0;
            static int cnt = 0;
            sum_dt += td.toc();
            cnt++;
            printf("visual front end cost: %f\n", sum_dt / cnt);
        }
#endif

        estimator->inputFeature(*curr);

        // update refer frame
        refer = curr;
    }

    // update prev frame
    prev = curr;

    // visualization
    VisualizationImg visualImg;
    visualImg.timeStamp = curr->timeStamp;
    visualImg.frameID = curr->frameID;
    visualImg.depthImg = curr->depthImg;
    visualImg.pointImg = pointTracker->drawPoints(curr, false);
    visualizer->inputImg(visualImg);

    LOG_EVERY_N(INFO, logFreq) << "Tracker inference cost: " << td.toc() << "ms";
}

void Tracker::trackStereoFusion(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &rightImg, const cv::Mat &depthImg,
                                const cv::Mat &p2cImg, const cv::Mat &l2rImg, const std::vector<std::pair<double, Eigen::Vector3d>> &gyr)
{
    std::unique_lock<std::mutex> lock(mTrack);

    utility::TicToc td;
    frameCnt++;

    curr = std::make_shared<Frame>(timeStamp, leftImg, rightImg, depthImg, p2cImg, l2rImg);

    // set lk optic flow predict by gyr
    if (!gyr.empty() && prev && !prev->points.empty())
    {
        double dt = curr->timeStamp - prev->timeStamp;
        Eigen::Matrix3d Ric = TicLeft.block<3, 3>(0, 0);
        Eigen::Matrix3d rot = utility::getRotationByImu(dt, Ric, gyr);  // prev to curr
        pointTracker->setPredictPoints(prev, rot);
    }

    // control frame freq
    // klt track points
    if (frameCnt % params.freqRatio != 0)
        pointTracker->inference(prev, curr, false);
    else
    {
        // extract points, lines and planes only when publish frame
        pointTracker->inference(prev, curr, true);

        estimator->inputFeature(*curr);

        // update refer frame
        refer = curr;
    }

    // update prev frame
    prev = curr;

    // visualization
    VisualizationImg visualImg;
    visualImg.timeStamp = curr->timeStamp;
    visualImg.frameID = curr->frameID;
    visualImg.depthImg = curr->depthImg;
    visualImg.pointImg = pointTracker->drawPoints(curr, false);

    visualizer->inputImg(visualImg);

    LOG_EVERY_N(INFO, logFreq) << "Tracker inference cost: " << td.toc() << "ms";
}

void Tracker::trackMono(const double timeStamp, const cv::Mat &leftImg, const std::vector<std::pair<double, Eigen::Vector3d>> &gyr)
{
    std::unique_lock<std::mutex> lock(mTrack);

    utility::TicToc td;
    frameCnt++;

    curr = std::make_shared<Frame>(timeStamp, leftImg);

    // set lk optic flow predict by gyr
    if (!gyr.empty() && prev && !prev->points.empty())
    {
        double dt = curr->timeStamp - prev->timeStamp;
        Eigen::Matrix3d Ric = TicLeft.block(0, 0, 3, 3);
        Eigen::Matrix3d rot = utility::getRotationByImu(dt, Ric, gyr);  // prev to curr
        pointTracker->setPredictPoints(prev, rot);
    }

    // control frame freq
    // klt track points
    if (frameCnt % params.freqRatio != 0)
        pointTracker->inference(prev, curr, false);
    else
    {
        // extract points, lines and planes only when publish frame
        pointTracker->inference(prev, curr, true);

        estimator->inputFeature(*curr);

        // update refer frame
        refer = curr;
    }

    // update prev frame
    prev = curr;

    // visualization
    VisualizationImg visualImg;
    visualImg.timeStamp = curr->timeStamp;
    visualImg.frameID = curr->frameID;
    visualImg.depthImg = curr->depthImg;
    visualImg.pointImg = pointTracker->drawPoints(curr, false);
    visualizer->inputImg(visualImg);

    LOG_EVERY_N(INFO, logFreq) << "Tracker inference cost: " << td.toc() << "ms";
}

void Tracker::trackMonoDepth(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &depthImg,
                             const std::vector<std::pair<double, Eigen::Vector3d>> &gyr)
{
    std::unique_lock<std::mutex> lock(mTrack);

    utility::TicToc td;
    frameCnt++;

    curr = std::make_shared<Frame>(timeStamp, leftImg, depthImg);

    // set lk optic flow predict by gyr
    if (!gyr.empty() && prev && !prev->points.empty())
    {
        double dt = curr->timeStamp - prev->timeStamp;
        Eigen::Matrix3d Ric = TicLeft.block(0, 0, 3, 3);
        Eigen::Matrix3d rot = utility::getRotationByImu(dt, Ric, gyr);  // prev to curr
        pointTracker->setPredictPoints(prev, rot);
    }

    // control frame freq
    // klt track points
    if (frameCnt % params.freqRatio != 0)
        pointTracker->inference(prev, curr, false);
    else
    {
        // extract points, lines and planes only when publish frame
        pointTracker->inference(prev, curr, true);

        estimator->inputFeature(*curr);

        // update refer frame
        refer = curr;
    }

    // update prev frame
    prev = curr;

    // visualization
    VisualizationImg visualImg;
    visualImg.timeStamp = curr->timeStamp;
    visualImg.frameID = curr->frameID;
    visualImg.depthImg = curr->depthImg;
    visualImg.pointImg = pointTracker->drawPoints(curr, false);
    visualizer->inputImg(visualImg);

    LOG_EVERY_N(INFO, logFreq) << "Tracker inference cost: " << td.toc() << "ms";
}

void Tracker::trackMonoOF(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &p2cImg,
                          const std::vector<std::pair<double, Eigen::Vector3d>> &gyr)
{
    std::unique_lock<std::mutex> lock(mTrack);

    utility::TicToc td;
    frameCnt++;

    curr = std::make_shared<Frame>(timeStamp, leftImg, p2cImg);

    // set lk optic flow predict by gyr
    if (!gyr.empty() && prev && !prev->points.empty())
    {
        double dt = curr->timeStamp - prev->timeStamp;
        Eigen::Matrix3d Ric = TicLeft.block(0, 0, 3, 3);
        Eigen::Matrix3d rot = utility::getRotationByImu(dt, Ric, gyr);  // prev to curr
        pointTracker->setPredictPoints(prev, rot);
    }

    // control frame freq
    // klt track points
    if (frameCnt % params.freqRatio != 0)
        pointTracker->inference(prev, curr, false);
    else
    {
        // extract points, lines and planes only when publish frame
        pointTracker->inference(prev, curr, true);

        estimator->inputFeature(*curr);

        // update refer frame
        refer = curr;
    }

    // update prev frame
    prev = curr;

    // visualization
    VisualizationImg visualImg;
    visualImg.timeStamp = curr->timeStamp;
    visualImg.frameID = curr->frameID;
    visualImg.depthImg = curr->depthImg;
    visualImg.pointImg = pointTracker->drawPoints(curr, false);
    visualizer->inputImg(visualImg);

    LOG_EVERY_N(INFO, logFreq) << "Tracker inference cost: " << td.toc() << "ms";
}

void Tracker::trackMonoFusion(const double timeStamp, const cv::Mat &leftImg, const cv::Mat &depthImg, const cv::Mat &p2cImg,
                              const std::vector<std::pair<double, Eigen::Vector3d>> &gyr)
{
    std::unique_lock<std::mutex> lock(mTrack);

    utility::TicToc td;
    frameCnt++;

    curr = std::make_shared<Frame>(timeStamp, leftImg, depthImg, p2cImg);

    // set lk optic flow predict by gyr
    if (!gyr.empty() && prev && !prev->points.empty())
    {
        double dt = curr->timeStamp - prev->timeStamp;
        Eigen::Matrix3d Ric = TicLeft.block(0, 0, 3, 3);
        Eigen::Matrix3d rot = utility::getRotationByImu(dt, Ric, gyr);  // prev to curr
        pointTracker->setPredictPoints(prev, rot);
    }

    // control frame freq
    // klt track points
    if (frameCnt % params.freqRatio != 0)
        pointTracker->inference(prev, curr, false);
    else
    {
        // extract points, lines and planes only when publish frame
        pointTracker->inference(prev, curr, true);

        estimator->inputFeature(*curr);

        // update refer frame
        refer = curr;
    }

    // update prev frame
    prev = curr;

    // visualization
    VisualizationImg visualImg;
    visualImg.timeStamp = curr->timeStamp;
    visualImg.frameID = curr->frameID;
    visualImg.depthImg = curr->depthImg;
    visualImg.pointImg = pointTracker->drawPoints(curr, false);
    visualizer->inputImg(visualImg);

    LOG_EVERY_N(INFO, logFreq) << "Tracker inference cost: " << td.toc() << "ms";
}

}  // namespace core
