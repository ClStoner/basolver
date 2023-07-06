#include "utils/Params.h"

Params params;

ParamManager::ParamManager(const std::string &configPath)
{
    LOG(INFO) << "ParamManager create";
    YAML::Node config = YAML::LoadFile(configPath);
    std::string pkgPath = getPkgPath(configPath);
    LOG(INFO) << "ros package path: " << pkgPath;

    // system params
    params.resultPath = pkgPath + config["resultPath"].as<std::string>();
    params.logLevel = config["logLevel"].as<int>();
    params.logFreq = config["logFreq"].as<int>();
    params.saveTrajectory = config["saveTrajectory"].as<bool>();
    params.vioTrajectory = config["vioTrajectory"].as<std::string>();
    params.loopTrajectory = config["loopTrajectory"].as<std::string>();
    if (params.saveTrajectory)
    {
        std::fstream vioTrajectoryFS{params.resultPath + "/" + params.vioTrajectory + ".txt", std::ios::out};
        vioTrajectoryFS.close();
        std::fstream loopTrajectoryFS{params.resultPath + "/" + params.loopTrajectory + ".txt", std::ios::out};
        loopTrajectoryFS.close();
    }

    // multi sensor relative params
    params.Stereo = config["Stereo"].as<bool>();
    params.Depth = config["Depth"].as<bool>();
    params.OpticalFlow = config["OpticalFlow"].as<bool>();
    params.syncTh = config["syncTh"].as<double>();
    params.freqRatio = config["freqRatio"].as<int>();

    // ros subscriber topic
    params.leftImgTopic = config["leftImgTopic"].as<std::string>();
    params.rightImgTopic = config["rightImgTopic"].as<std::string>();
    params.depthImgTopic = config["depthImgTopic"].as<std::string>();
    params.l2rOFImgTopic = config["l2rOFImgTopic"].as<std::string>();
    params.p2cOFImgTOpic = config["p2cOFImgTOpic"].as<std::string>();
    params.imuTopic = config["imuTopic"].as<std::string>();

    // img relative params
    params.leftCalibPath = pkgPath + config["leftCalibPath"].as<std::string>();
    params.leftCam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(params.leftCalibPath);
    if (params.Stereo)
    {
        params.rightCalibPath = pkgPath + config["rightCalibPath"].as<std::string>();
        params.rightCam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(params.rightCalibPath);
    }

    params.imgHeight = config["imgHeight"].as<int>();
    params.imgWidth = config["imgWidth"].as<int>();
    params.depthScale = 1.0 / config["depthScale"].as<double>();
    params.depthTh = config["depthTh"].as<double>();
    params.focalLength = 460.0;
    params.baseLine = config["baseLine"].as<double>();
    params.clahe = config["clahe"].as<bool>();
    params.borderSize = config["borderSize"].as<int>();

    this->readIntrinsics(params.leftCalibPath);

    // extrinsic from camera to imu
    params.estimateExtrinsic = config["estimateExtrinsic"].as<bool>();
    if (params.Stereo)
    {
        CHECK_EQ(config["TicLeft"].IsSequence(), true);
        CHECK_EQ(config["TicRight"].IsSequence(), true);
        CHECK_EQ(config["TicLeft"].size(), 16);
        CHECK_EQ(config["TicRight"].size(), 16);
        params.TicLeft = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(config["TicLeft"].as<std::vector<double>>().data(), 4, 4);
        params.TicRight = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(config["TicRight"].as<std::vector<double>>().data(), 4, 4);
    }
    else
    {
        CHECK_EQ(config["TicLeft"].IsSequence(), true);
        CHECK_EQ(config["TicLeft"].size(), 16);
        params.TicLeft = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(config["TicLeft"].as<std::vector<double>>().data(), 4, 4);
    }

    if (params.baseLine > 0)
    {
        CHECK_EQ(config["TicLeft"].IsSequence(), true);
        CHECK_EQ(config["TicLeft"].size(), 16);
        params.TicLeft = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(config["TicLeft"].as<std::vector<double>>().data(), 4, 4);
        Eigen::Matrix4d T_Left_Right = Eigen::Matrix4d::Identity();
        T_Left_Right(0, 3) = params.baseLine;
        params.TicRight = params.TicLeft * T_Left_Right;
    }

    // imu noise params
    params.accN = config["accN"].as<double>();
    params.accW = config["accW"].as<double>();
    params.gyrN = config["gyrN"].as<double>();
    params.gyrW = config["gyrW"].as<double>();
    params.gNorm = config["gNorm"].as<double>();
    params.G = Eigen::Vector3d{0, 0, params.gNorm};

    // imu td params
    params.td = config["td"].as<double>();
    params.estimateTD = config["estimateTD"].as<bool>();

    // estimator params
    params.maxSolverTime = config["maxSolverTime"].as<double>();
    params.maxSolverIterations = config["maxSolverIterations"].as<int>();
    params.keyframeParallax = config["keyframeParallax"].as<double>() / params.focalLength;
    params.outlierReprojectTh = config["outlierReprojectTh"].as<double>();

    // loop relative params
    params.useLoop = config["useLoop"].as<bool>();

    // point feature params
    params.pointExtractType = config["pointExtractType"].as<int>();
    params.pointExtractMaxNum = config["pointExtractMaxNum"].as<int>();
    params.pointExtractMinDist = config["pointExtractMinDist"].as<int>();
    params.pointMatchRansacTh = config["pointMatchRansacTh"].as<double>();

    // init glog and print params
    initDirs(params.resultPath);
    initLog();
    printParams();

    return;
}

void ParamManager::readIntrinsics(const std::string &calibPath)
{
    YAML::Node config = YAML::LoadFile(calibPath);
    if (params.leftCam->modelType() == camodocal::Camera::PINHOLE)
    {
        double fx = config["projection_parameters"]["fx"].as<double>();
        double fy = config["projection_parameters"]["fy"].as<double>();
        double cx = config["projection_parameters"]["cx"].as<double>();
        double cy = config["projection_parameters"]["cy"].as<double>();

        double k1 = config["distortion_parameters"]["k1"].as<double>();
        double k2 = config["distortion_parameters"]["k2"].as<double>();
        double p1 = config["distortion_parameters"]["p1"].as<double>();
        double p2 = config["distortion_parameters"]["p2"].as<double>();
        params.K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        params.D = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
        // cv::Mat newK = (cv::Mat_<double>(3, 3) << focalLength, 0, imgWidth / 2, 0, focalLength, imgHeight / 2, 0, 0, 1);
        params.leftCam->initUndistortRectifyMap(params.undistX, params.undistY);
    }
    else
    {
        params.K = (cv::Mat_<double>(3, 3) << 1, 0, 1, 0, 0, 0, 0, 0, 1);
        params.D = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
    }
}

void ParamManager::printParams()
{
    LOG(INFO) << "ParamManager print";

    // system params
    LOG(INFO) << "resultPath: " << params.resultPath;
    LOG_IF(INFO, params.logLevel == 0) << "logLevel: INFO";
    LOG_IF(INFO, params.logLevel == 1) << "logLevel: WARNING";
    LOG_IF(INFO, params.logLevel == 2) << "logLevel: ERROR";
    LOG_IF(INFO, params.logLevel == 3) << "logLevel: FATAL";
    LOG(INFO) << "logFreq: " << params.logFreq;
    LOG(INFO) << "saveTrajectory: " << params.saveTrajectory;
    LOG(INFO) << "vioTrajectory: " << params.vioTrajectory;
    LOG(INFO) << "loopTrajectory: " << params.loopTrajectory;

    // multi sensor relative params
    LOG(INFO) << "Stereo: " << params.Stereo;
    LOG(INFO) << "Depth: " << params.Depth;
    LOG(INFO) << "syncTh: " << params.syncTh;
    LOG(INFO) << "freqRatio: " << params.freqRatio;

    // ros subscriber topic
    LOG(INFO) << "leftImgTopic: " << params.leftImgTopic;
    LOG(INFO) << "rightImgTopic: " << params.rightImgTopic;
    LOG(INFO) << "depthImgTopic: " << params.depthImgTopic;
    LOG(INFO) << "l2rOFImgTopic: " << params.l2rOFImgTopic;
    LOG(INFO) << "p2cOFImgTOpic: " << params.p2cOFImgTOpic;
    LOG(INFO) << "imuTopic: " << params.imuTopic;

    // img relative params
    LOG(INFO) << "leftCalibPath: " << params.leftCalibPath;
    LOG(INFO) << "rightCalibPath: " << params.rightCalibPath;
    LOG(INFO) << "imgHeight: " << params.imgHeight;
    LOG(INFO) << "imgWidth: " << params.imgWidth;
    LOG(INFO) << "depthScale: " << params.depthScale;
    LOG(INFO) << "depthTh: " << params.depthTh;
    LOG(INFO) << "focalLength: " << params.focalLength;
    LOG(INFO) << "baseLine: " << params.baseLine;
    LOG(INFO) << "clahe: " << params.clahe;
    LOG(INFO) << "borderSize: " << params.borderSize;
    // LOG(INFO) << "projection_parameters: \n" << params.K;
    // LOG(INFO) << "distortion_parameters: " << params.D.t();

    // extrinsic from camera to imu
    LOG(INFO) << "estimateExtrinsic: " << params.estimateExtrinsic;
    LOG(INFO) << "TicLeft: \n" << params.TicLeft;
    LOG(INFO) << "TicRight: \n" << params.TicRight;

    // imu noise params
    LOG(INFO) << "accN: " << params.accN;
    LOG(INFO) << "accW: " << params.accW;
    LOG(INFO) << "gyrN: " << params.gyrN;
    LOG(INFO) << "gyrW: " << params.gyrW;
    LOG(INFO) << "gNorm: " << params.gNorm;
    LOG(INFO) << "G: " << params.G.transpose();

    // imu td params
    LOG(INFO) << "td: " << params.td;
    LOG(INFO) << "estimateTD: " << params.estimateTD;

    // estimator params
    LOG(INFO) << "maxSolverTime: " << params.maxSolverTime;
    LOG(INFO) << "maxSolverIterations: " << params.maxSolverIterations;
    LOG(INFO) << "keyframeParallax: " << params.keyframeParallax;
    LOG(INFO) << "outlierReprojectTh: " << params.outlierReprojectTh;

    // loop relative params
    LOG(INFO) << "useLoop: " << params.useLoop;

    // point feature params
    LOG_IF(INFO, params.pointExtractType == PointExtractFlag::FAST) << "pointExtractType: Fast";
    LOG_IF(INFO, params.pointExtractType == PointExtractFlag::Harris) << "pointExtractType: Harris";
    LOG_IF(INFO, params.pointExtractType == PointExtractFlag::None) << "pointExtractType: Origin";
    LOG_IF(ERROR, params.pointExtractType < 0 || params.pointExtractType > 2) << "pointExtractType: Origin";

    LOG(INFO) << "pointExtractMaxNum: " << params.pointExtractMaxNum;
    LOG(INFO) << "pointExtractMinDist: " << params.pointExtractMinDist;
}

std::string ParamManager::getPkgPath(const std::string &configPath)
{
    int pn = configPath.find_last_of('/');
    std::string packagePath = configPath.substr(0, pn);
    pn = packagePath.find_last_of('/');
    packagePath = packagePath.substr(0, pn);
    pn = packagePath.find_last_of('/');
    packagePath = packagePath.substr(0, pn) + '/';

    return packagePath;
}

void ParamManager::initDirs(const std::string &resultPath)
{
    LOG(INFO) << "init result dir";
    std::string logPath = resultPath + "/log";
    std::string imgPath = resultPath + "/img";
    if (access(resultPath.c_str(), 0) != 0)
    {
        LOG(WARNING) << "mkdir result dir: " << resultPath;
        mkdir(resultPath.c_str(), S_IRWXU);
    }

    if (access(logPath.c_str(), 0) != 0)
    {
        LOG(WARNING) << "mkdir log dir: " << logPath;
        mkdir(logPath.c_str(), S_IRWXU);
    }

    if (access(imgPath.c_str(), 0) != 0)
    {
        LOG(WARNING) << "mkdir img dir: " << imgPath;
        mkdir(imgPath.c_str(), S_IRWXU);
    }
}

void ParamManager::initLog()
{
    // Glog
    FLAGS_log_dir = params.resultPath + "/log";
    FLAGS_colorlogtostderr = true;
    google::SetStderrLogging(params.logLevel);
    google::SetLogFilenameExtension("log");
    google::InitGoogleLogging("VINS_MHD");
    LOG(INFO) << "Glog register";

    if (params.saveTrajectory)
    {
        std::fstream vioTrajectoryFS{params.resultPath + "/" + params.vioTrajectory + ".txt", std::ios::out};
        vioTrajectoryFS.close();

        std::fstream loopTrajectoryFS{params.resultPath + "/" + params.loopTrajectory + ".txt", std::ios::out};
        loopTrajectoryFS.close();

        std::fstream debugFS{params.resultPath + "/" + "debug.txt", std::ios::out};
        debugFS.close();
    }
}
