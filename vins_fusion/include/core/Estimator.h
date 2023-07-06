/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "utils/Marco.h"
#include "utils/Utility.h"
#include "utils/Visualization.h"
#include "initial/Solve5pts.h"
#include "initial/InitialSFM.h"
#include "initial/InitialAlignment.h"
#include "initial/InitialExRotation.h"
#include "factor/IMUFactor.h"
#include "factor/PoseLocalParameterization.h"
#include "factor/MarginalizationFactor.h"
#include "factor/ProjectionTwoFrameOneCamFactor.h"
#include "factor/ProjectionTwoFrameTwoCamFactor.h"
#include "factor/ProjectionOneFrameTwoCamFactor.h"
#include "core/FeatureManager.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>
#include "../../basolver/basolver.h"

#include <thread>
#include <tuple>

namespace core
{
class Estimator
{
  public:
    enum SolverFlag
    {
        INITIAL = 0,
        NON_LINEAR = 1
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    POINTER_TYPEDEFS(Estimator);
    Estimator(const Visualization::Ptr &visualizer);

    ~Estimator();

    // initial params
    void clearState();

    void setParameter();

    void setConfigFile(std::string config_file, std::string init_file)
    {
        LOG(INFO) << "BaSolver setConfigFile: " << config_file;
        BaSolver::readParameters(config_file);
        init_config_file_ = init_file;
        config_file_ = config_file;
        LOG(INFO) << "BaSolver readParameters";
    }

    // interface
    void inputIMU(double t, const Eigen::Vector3d &linearAcceleration, const Eigen::Vector3d &angularVelocity);

    void inputFeature(const Frame &frame);

  private:
    // main function
    void processMeasurements();

    void processIMU(double t, double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);

    void processImage(const double timeStamp, const Frame &feature);

    void visualization(const double timeStamp, const long frameID);

    void pubKeyframe();

    // internal fucntion
    // preprocess
    bool imuAvailable(double t);

    bool getIMUInterval(double t0, double t1, std::vector<std::pair<double, Eigen::Vector3d>> &accVector,
                        std::vector<std::pair<double, Eigen::Vector3d>> &gyrVector);

    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);

    bool getMeasurements();

    // initialization
    bool monoInitialization();

    bool stereoInitialization();

    bool monoVisualInitialAlign();

    bool stereoVisualInitialAlign();

    bool relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);

    // slide window
    void slideWindow();

    void slideWindowNew();

    void slideWindowOld();

    void updateLatestStates();

    // ceres BA
    void optimization();

    void originVector2Double();

    void originDouble2Vector();

    // process outlier
    bool failureDetection();

    double outliersRejection(std::set<long> &removeIndex);

    void removeOutliers(const int pointNum);

    double reprojectionError(Eigen::Matrix3d &Ri, Eigen::Vector3d &Pi, Eigen::Matrix3d &rici, Eigen::Vector3d &tici, Eigen::Matrix3d &Rj,
                             Eigen::Vector3d &Pj, Eigen::Matrix3d &ricj, Eigen::Vector3d &ticj, double depth, Eigen::Vector3d &uvi,
                             Eigen::Vector3d &uvj);

    // base params
    const int logFreq;
    const int windowSize;
    const bool stereo;
    const int cameraNum;
    const double focalLength;
    const double maxSolverTime;
    const int maxSolverIterations;
    const double outlierReprojectTh;
    const bool estimateExtrinsic;
    const bool estimateTD;
    const bool useLoop;

    // internal worker
    MotionEstimator motionEstimator;
    InitialEXRotation initialExRotation;
    InitialAlignment initialAligment;
    FeatureManager featureManager;

    // external worker
    Visualization::Ptr visualizer;

    // estimator input data queue
    std::condition_variable con;
    std::mutex mBuf;        // feature and imu queue mutex
    std::mutex mProcess;    // frame process mutex
    std::mutex mPropagate;  // imu fast propagate mutex
    std::queue<std::pair<double, Eigen::Vector3d>> accBuf;
    std::queue<std::pair<double, Eigen::Vector3d>> gyrBuf;
    std::queue<Frame> featureBuf;

    // estimator flag
    SolverFlag solverFlag;
    MarginalizationFlag marginalizationFlag;

    // estimator internal notion variable
    double prevTime, curTime;
    bool openExEstimation;
    bool firstIMU;
    bool failureOccur;
    double initialTimeStamp;
    int frameCount;

    // imu propagate state for fast response
    double latestTime;
    Eigen::Vector3d latestP, latestV, latestBa, latestBg, latestAcc_0, latestGyr_0;
    Eigen::Quaterniond latestQ;

    // keyframe state for publish
    double updateTime;
    Eigen::Vector3d updateP, updateV;
    Eigen::Quaterniond updateQ;

    // save state for slide window optimazation
    Eigen::Matrix3d backR0, lastR, lastR0;
    Eigen::Vector3d backP0, lastP, lastP0;

    // save latest imu data for imu pre integration
    Eigen::Vector3d acc_0, gyr_0;
    std::vector<Eigen::Vector3d> linearAccBuf[params.windowSize + 1];
    std::vector<Eigen::Vector3d> angularVelBuf[params.windowSize + 1];

    // main thread deal with input feature and imu data
    std::thread processThread;

    // extrinsics params
    std::vector<Eigen::Matrix3d> Ric;
    std::vector<Eigen::Vector3d> tic;
    Eigen::Matrix3d Rl2r;
    Eigen::Vector3d tl2r;

    // slide window
    double Headers[params.windowSize + 1];
    Frame Frames[params.windowSize + 1];
    Eigen::Vector3d Ps[params.windowSize + 1];
    Eigen::Vector3d Vs[params.windowSize + 1];
    Eigen::Matrix3d Rs[params.windowSize + 1];
    Eigen::Vector3d Bas[params.windowSize + 1];
    Eigen::Vector3d Bgs[params.windowSize + 1];
    std::vector<double> dt_buf[params.windowSize + 1];
    double td;
    Eigen::Vector3d g;

    // variables for imu pre integration and marginalization
    BaSolver::IntegrationBase *tmp_pre_integration_;
    BaSolver::IntegrationBase *pre_integrations[params.windowSize + 1];
    // MarginalizationInfo::Ptr last_marginalization_info;
    // std::vector<double *> last_marginalization_parameter_blocks;

    // ceres optimazation variables
    double paraPose[params.windowSize + 1][SIZE_POSE];
    double paraSpeedBias[params.windowSize + 1][SIZE_SPEEDBIAS];
    double paraFeature[params.featureNum][SIZE_POINT];

    double paraExPose[2][SIZE_POSE];
    double paraTd[1][1];

    // save all estimator frame
    std::map<double, ImageFrame> allImageFrame;

    BaSolver::MarginalizationInfo *ba_last_marginalization_info;
    vector<double *> ba_last_marginalization_parameter_blocks;
    Eigen::Matrix2d project_sqrt_info_;

    

    std::string config_file_;
    std::string init_config_file_;

};

}  // namespace core
