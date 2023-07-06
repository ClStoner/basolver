#pragma once

#include "Marco.h"

#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/PinholeCamera.h>
#include <camodocal/camera_models/CataCamera.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <sys/stat.h>
#include <sys/types.h>

struct Params
{
    // system params
    std::string resultPath;
    int logLevel;
    int logFreq;
    bool saveTrajectory;
    std::string vioTrajectory;
    std::string loopTrajectory;

    // ros subscriber topic
    std::string leftImgTopic, rightImgTopic;
    std::string depthImgTopic;
    std::string l2rOFImgTopic, p2cOFImgTOpic;
    std::string imuTopic;

    // img relative params
    std::string leftCalibPath, rightCalibPath;
    camodocal::CameraPtr leftCam, rightCam;
    int imgHeight;
    int imgWidth;
    double depthScale;
    double depthTh;
    double focalLength;
    double baseLine;
    bool clahe;
    int borderSize;

    cv::Mat K, D;
    cv::Mat undistX, undistY;

    // multi sensor relative params
    bool Stereo;
    bool Depth;
    bool OpticalFlow;
    double syncTh;
    int freqRatio;

    // extrinsic from camera to imu
    Eigen::Matrix4d TicLeft;
    Eigen::Matrix4d TicRight;
    bool estimateExtrinsic;

    // imu noise params
    double accN, accW;
    double gyrN, gyrW;
    double gNorm;
    Eigen::Vector3d G;

    // imu td params
    double td;
    bool estimateTD;

    // estimator params
    static constexpr int windowSize = 10;
    static constexpr int featureNum = 1000;
    static constexpr double FOCAL_LENGTH = 460.0;
    const double initDepth = 5.0;
    double maxSolverTime;
    int maxSolverIterations;
    double keyframeParallax;
    double outlierReprojectTh;

    // loop relative params
    bool useLoop;

    // point feature params
    int pointExtractType;
    int pointExtractMaxNum;
    int pointExtractMinDist;
    double pointMatchRansacTh;
};

class ParamManager
{
  public:
    POINTER_TYPEDEFS(ParamManager);
    ParamManager(const std::string &configPath);

  private:
    void readIntrinsics(const std::string &calibPath);

    std::string getPkgPath(const std::string &configPath);

    void initDirs(const std::string &pkgPath);

    void initLog();

    void printParams();
};

extern Params params;
