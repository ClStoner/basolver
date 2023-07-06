/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "core/Estimator.h"

namespace core
{
Estimator::Estimator(const Visualization::Ptr &visualizer)
    : logFreq(params.logFreq), windowSize(params.windowSize), stereo(params.Stereo || params.Depth), cameraNum(1 + stereo),
      focalLength(params.focalLength), maxSolverTime(params.maxSolverTime), maxSolverIterations(params.maxSolverIterations),
      outlierReprojectTh(params.outlierReprojectTh), estimateExtrinsic(params.estimateExtrinsic), estimateTD(params.estimateTD),
      useLoop(params.useLoop)
{
    LOG(INFO) << "Estimator create";
    this->clearState();

    // extern worker
    this->visualizer = visualizer;
}

Estimator::~Estimator()
{
    processThread.join();
    LOG(INFO) << "Estimator destroy, join thread";
}

void Estimator::clearState()
{
    LOG(INFO) << "Estimator clearState";

    std::unique_lock<std::mutex> lock(mProcess);

    // clear feature and imu queue
    std::queue<std::pair<double, Eigen::Vector3d>>().swap(accBuf);
    std::queue<std::pair<double, Eigen::Vector3d>>().swap(gyrBuf);
    std::queue<Frame>().swap(featureBuf);

    solverFlag = SolverFlag::INITIAL;
    prevTime = -1;
    curTime = 0;
    openExEstimation = false;
    firstIMU = false;
    failureOccur = 0;
    initialTimeStamp = 0;
    frameCount = 0;

    for (int i = 0; i < windowSize + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linearAccBuf[i].clear();
        angularVelBuf[i].clear();
        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }
    Ric.resize(cameraNum);
    tic.resize(cameraNum);

    allImageFrame.clear();

    // if (last_marginalization_info)
    //     last_marginalization_info = nullptr;

    if (tmp_pre_integration_ != nullptr)
        delete tmp_pre_integration_;
    
    if(ba_last_marginalization_info != nullptr)
        delete ba_last_marginalization_info;

    tmp_pre_integration_ = nullptr;

    ba_last_marginalization_info = nullptr;
    ba_last_marginalization_parameter_blocks.clear();


    // last_marginalization_parameter_blocks.clear();

    featureManager.clearState();
}

void Estimator::setParameter()
{
    LOG(INFO) << "Estimator setParameter";

    std::unique_lock<std::mutex> lock(mProcess);

    Ric[0] = params.TicLeft.block(0, 0, 3, 3);
    tic[0] = params.TicLeft.block(0, 3, 3, 1);
    LOG(INFO) << "left exitrinsic R: \n" << Ric[0];
    LOG(INFO) << "left exitrinsic t: \n" << tic[0].transpose();
    if (stereo)
    {
        Ric[1] = params.TicRight.block(0, 0, 3, 3);
        tic[1] = params.TicRight.block(0, 3, 3, 1);
        LOG(INFO) << "right exitrinsic R: \n" << Ric[1];
        LOG(INFO) << "right exitrinsic t: \n" << tic[1].transpose();

        // calc stereo extrinsics
        Eigen::Matrix4d Tl2r = params.TicRight.inverse() * params.TicLeft;
        Rl2r = Tl2r.block(0, 0, 3, 3);
        tl2r = Tl2r.block(0, 3, 3, 1);
        LOG(INFO) << "Rl2r: " << utility::R2ypr(Rl2r).transpose();
        LOG(INFO) << "tl2r: " << tl2r.transpose();
    }

    ProjectionTwoFrameOneCamFactor::sqrt_info = params.focalLength / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = params.focalLength / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = params.focalLength / 1.5 * Matrix2d::Identity();
    // LineProjectionFactorTwoFrame::sqrt_info = params.focalLength / 1.5 * Matrix2d::Identity();

    project_sqrt_info_ = params.FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = params.td;
    g = Eigen::Vector3d{0, 0, params.gNorm};
    LOG(INFO) << "set g " << g.transpose();

    processThread = std::thread(&Estimator::processMeasurements, this);
}

void Estimator::inputIMU(double t, const Eigen::Vector3d &linearAcceleration, const Eigen::Vector3d &angularVelocity)
{
    std::unique_lock<std::mutex> lock1(mBuf);
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    con.notify_one();

    if (solverFlag == NON_LINEAR)
        fastPredictIMU(t, linearAcceleration, angularVelocity);
}

void Estimator::inputFeature(const Frame &frame)
{
    std::unique_lock<std::mutex> lock(mBuf);
    featureBuf.push(frame);
    con.notify_one();
}

bool Estimator::getIMUInterval(double t0, double t1, std::vector<std::pair<double, Eigen::Vector3d>> &accVector,
                               std::vector<std::pair<double, Eigen::Vector3d>> &gyrVector)
{
    if (accBuf.empty())
    {
        LOG(WARNING) << "not receive imu";
        return false;
    }

    if (t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            gyrVector.push_back(gyrBuf.front());
            accBuf.pop();
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        LOG(WARNING) << "wait for imu";
        return false;
    }

    return true;
}

bool Estimator::imuAvailable(double t)
{
    if (!accBuf.empty() && !gyrBuf.empty() && t <= accBuf.back().first && t <= gyrBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (true)
    {
        // deal with feature and imu data queue
        std::unique_lock<std::mutex> lock1(mBuf);
        con.wait(lock1, [&] { return getMeasurements(); });

        utility::TicToc dt;

        Frame feature = std::move(featureBuf.front());
        curTime = feature.timeStamp + td;
        featureBuf.pop();

        std::vector<std::pair<double, Eigen::Vector3d>> accVector, gyrVector;
        getIMUInterval(prevTime, curTime, accVector, gyrVector);
        lock1.unlock();

        // process IMU
        for (unsigned i = 0; i < accVector.size(); i++)
        {
            double dt;
            if (i == 0)
                dt = accVector[i].first - prevTime;  // first imu data
            else if (i == accVector.size() - 1)
                dt = curTime - accVector[i - 1].first;  // end imu data
            else
                dt = accVector[i].first - accVector[i - 1].first;  // middle imu data

            processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
        }

        // process input feature frame
        processImage(feature.timeStamp, feature);
        prevTime = curTime;

        if (useLoop)
        {
            pubKeyframe();
        }

        visualization(feature.timeStamp, feature.frameID);

        LOG_EVERY_N(INFO, logFreq) << "estimator process frame cost: " << dt.toc();
    }
}

void Estimator::pubKeyframe()
{
    if (solverFlag == SolverFlag::NON_LINEAR && marginalizationFlag == MarginalizationFlag::MARGIN_OLD)
    {
        Keyframe keyframe;

        const int keyframeIdx = windowSize - 2;
        keyframe.timeStamp = Headers[keyframeIdx];
        keyframe.Ric = Ric[0];
        keyframe.tic = tic[0];
        keyframe.R = Rs[keyframeIdx];
        keyframe.t = Ps[keyframeIdx];

        for (auto &idFeature : featureManager.pointFeatures)
        {
            auto &featureID = idFeature.first;
            auto &feature = idFeature.second;
            if (feature.startFrame < windowSize - 2 && feature.endFrame() >= windowSize - 2 && feature.solveFlag == 1)
            {

                int imu_i = feature.startFrame;
                int imu_j = windowSize - 2 - feature.startFrame;

                Eigen::Vector3d pt_i = feature.featureFrames[0].point * feature.estimatedDepth;
                Eigen::Vector3d pt_w = Rs[imu_i] * (Ric[0] * pt_i + tic[0]) + Ps[imu_i];
                keyframe.pts3D.push_back(pt_w);
                keyframe.ptsUn.push_back(feature.featureFrames[imu_j].point);
                keyframe.ptsUv.push_back(feature.featureFrames[imu_j].uv);
                keyframe.ptsID.push_back(featureID);
            }
        }

        visualizer->inputKeyframe(keyframe);
    }
}

bool Estimator::getMeasurements()
{
    if (!featureBuf.empty() && !accBuf.empty() && !gyrBuf.empty())
    {
        double featureTimeStamp = featureBuf.front().timeStamp;
        return imuAvailable(featureTimeStamp + td);
    }
    else
        return false;
}

void Estimator::processIMU(double t, double dt, const Eigen::Vector3d &linearAcceleration, const Eigen::Vector3d &angularVelocity)
{
    if (!firstIMU)
    {
        firstIMU = true;
        acc_0 = linearAcceleration;
        gyr_0 = angularVelocity;
    }

    if (!pre_integrations[frameCount])
        pre_integrations[frameCount] = new BaSolver::IntegrationBase{acc_0, gyr_0, Bas[frameCount], Bgs[frameCount]};

    if (frameCount != 0)
    {
        pre_integrations[frameCount]->push_back(dt, linearAcceleration, angularVelocity);
        tmp_pre_integration_->push_back(dt, linearAcceleration, angularVelocity);

        dt_buf[frameCount].push_back(dt);
        linearAccBuf[frameCount].push_back(linearAcceleration);
        angularVelBuf[frameCount].push_back(angularVelocity);

        int j = frameCount;
        Eigen::Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angularVelocity) - Bgs[j];
        Rs[j] *= utility::deltaQ(un_gyr * dt).toRotationMatrix();

        Eigen::Vector3d un_acc_1 = Rs[j] * (linearAcceleration - Bas[j]) - g;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linearAcceleration;
    gyr_0 = angularVelocity;
}

void Estimator::processImage(const double timeStamp, const Frame &feature)
{    
    std::unique_lock<std::mutex> lock(mProcess);

    utility::TicToc dt;

    if (featureManager.addFeatureCheckParallax(frameCount, feature, td))
    {
        marginalizationFlag = MARGIN_OLD;
        // LOG(INFO) << "keyframe";
    }
    else
    {
        marginalizationFlag = MARGIN_SECOND_NEW;
        // LOG(INFO) << "non-keyframe";
    }

    // LOG(INFO)<<"%s", marginalizationFlag ? "Non-keyframe" : "Keyframe";
    // LOG(INFO)<<"Solving %d", frameCount;
    int pointsNum = featureManager.getPointCount();
    Headers[frameCount] = timeStamp;
    Frames[frameCount] = feature;
    ImageFrame imageframe(timeStamp, feature);
    imageframe.pre_integration = tmp_pre_integration_;
    allImageFrame.insert(make_pair(timeStamp, imageframe));
    tmp_pre_integration_ = new BaSolver::IntegrationBase{acc_0, gyr_0, Bas[frameCount], Bgs[frameCount]};
    if (solverFlag == INITIAL)
    {
        // mono/stereo + IMU initilization
        if (frameCount == windowSize)
        {
            bool result = false;
            if ((timeStamp - initialTimeStamp) > 0.1)
            {
                if (!stereo)
                    result = monoInitialization();
                else
                    result = stereoInitialization();

                initialTimeStamp = timeStamp;
            }

            if (result)
            {
                optimization();
                removeOutliers(pointsNum);
                updateLatestStates();
                solverFlag = NON_LINEAR;
                slideWindow();
            }
            else
                slideWindow();
        }

        if (frameCount < windowSize)
        {
            frameCount++;
            Ps[frameCount] = Ps[frameCount - 1];
            Vs[frameCount] = Vs[frameCount - 1];
            Rs[frameCount] = Rs[frameCount - 1];
            Bas[frameCount] = Bas[frameCount - 1];
            Bgs[frameCount] = Bgs[frameCount - 1];
        }
    }
    else
    {
        // slid window optimization
        featureManager.triangulate(frameCount, Ps, Rs, Ric, tic);

        optimization();
        removeOutliers(pointsNum);

        if (failureDetection())
        {
            LOG(WARNING) << "failure detection!";
            failureOccur = 1;
            clearState();
            setParameter();
            LOG(WARNING) << "system reboot!";
            return;
        }

        slideWindow();
        featureManager.removeFailures();

        lastR = Rs[windowSize];
        lastP = Ps[windowSize];
        lastR0 = Rs[0];
        lastP0 = Ps[0];
        updateLatestStates();
    }
}

void Estimator::visualization(const double timeStamp, const long frameID)
{
    // visualization latest odometry
    if (solverFlag == SolverFlag::NON_LINEAR)
    {
        VisualizationOdom visualOdom;
        visualOdom.timeStamp = timeStamp;
        visualOdom.frameID = frameID;
        visualOdom.P = Ps[windowSize];
        visualOdom.V = Vs[windowSize];
        visualOdom.R = Eigen::Quaterniond{Rs[windowSize]};
        for (unsigned i = 0; i < Ric.size(); i++)
        {
            visualOdom.Ric.emplace_back(Eigen::Quaterniond{Ric[i]});
            visualOdom.tic.emplace_back(tic[i]);
        }
        visualizer->inputOdom(visualOdom);

        // print vio path
        LOG_EVERY_N(WARNING, logFreq) << std::fixed << std::setprecision(4) << "td: " << td << " p: " << visualOdom.P.x() << " " << visualOdom.P.y()
                                      << " " << visualOdom.P.z() << " ypr: " << utility::R2ypr(visualOdom.R.toRotationMatrix()).x() << " "
                                      << utility::R2ypr(visualOdom.R.toRotationMatrix()).y() << " "
                                      << utility::R2ypr(visualOdom.R.toRotationMatrix()).z() << " ba: " << Bas[windowSize].transpose();

        // save vio path
        if (params.saveTrajectory)
        {
            std::fstream trajectoryFS{params.resultPath + "/" + params.vioTrajectory + ".txt", std::ios::app};
            trajectoryFS.setf(std::ios::fixed, std::ios::floatfield);
            trajectoryFS.precision(9);
            trajectoryFS << timeStamp << " ";
            trajectoryFS.precision(5);
            trajectoryFS << visualOdom.P.x() << " " << visualOdom.P.y() << " " << visualOdom.P.z() << " " << visualOdom.R.w() << " "
                         << visualOdom.R.x() << " " << visualOdom.R.y() << " " << visualOdom.R.z() << std::endl;
            trajectoryFS.close();
        }

        Visualization3D visual3D;
        visual3D.timeStamp = timeStamp;
        visual3D.frameID = frameID;

        // latest points
        for (auto &idFeature : featureManager.pointFeatures)
        {
            auto &feature = idFeature.second;
            int used_num = feature.featureFrames.size();
            if (!(used_num >= 2 && feature.startFrame < windowSize - 2))
                continue;

            if (feature.startFrame > windowSize * 3.0 / 4.0 || feature.solveFlag != 1)
                continue;

            int imu_i = feature.startFrame;
            Eigen::Vector3d p3DCamera = feature.featureFrames[0].point * feature.estimatedDepth;
            Eigen::Vector3d p3DWorld = Rs[imu_i] * (Ric[0] * p3DCamera + tic[0]) + Ps[imu_i];
            visual3D.pts3D.emplace_back(p3DWorld);
        }

        // margin points
        for (auto &idFeature : featureManager.pointFeatures)
        {
            auto &feature = idFeature.second;
            int used_num = feature.featureFrames.size();
            if (!(used_num >= 2 && feature.startFrame < windowSize - 2))
                continue;

            if (feature.startFrame == 0 && feature.featureFrames.size() <= 2 && feature.solveFlag == 1)
            {
                int imu_i = feature.startFrame;
                Eigen::Vector3d p3DCamera = feature.featureFrames[0].point * feature.estimatedDepth;
                Eigen::Vector3d p3DWorld = Rs[imu_i] * (Ric[0] * p3DCamera + tic[0]) + Ps[imu_i];
                visual3D.marginPts3D.emplace_back(p3DWorld);
            }
        }

        visualizer->inputFeature(visual3D);
    }
}

// mono init
bool Estimator::monoInitialization()
{
    // check imu observibility
    {
        std::map<double, ImageFrame>::iterator frame_it;
        Eigen::Vector3d sum_g;
        for (frame_it = allImageFrame.begin(), frame_it++; frame_it != allImageFrame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }

        Eigen::Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)allImageFrame.size() - 1);
        double var = 0;
        for (frame_it = allImageFrame.begin(), frame_it++; frame_it != allImageFrame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // LOG(INFO) << "frame g " << tmp_g.transpose() ;
        }
        var = sqrt(var / ((int)allImageFrame.size() - 1));
        // LOG(WARNING)<<("IMU variation %f!", var);
        if (var < 0.25)
        {
            LOG(INFO) << "IMU excitation not enouth!";
            // return false;
        }
    }

    // global sfm
    std::vector<SFMFeature> sfmFeatures;
    for (auto &idFeature : featureManager.pointFeatures)
    {
        auto &featureID = idFeature.first;
        auto &feature = idFeature.second;

        SFMFeature sfmFeature;
        sfmFeature.state = false;
        sfmFeature.id = featureID;
        int imu_j = feature.startFrame;
        for (auto &featureFrame : feature.featureFrames)
        {
            Eigen::Vector2d pts_j = featureFrame.point.head(2);
            sfmFeature.observation.emplace_back(imu_j, pts_j);
            imu_j++;
        }
        sfmFeatures.push_back(sfmFeature);
    }

    // calc findmental rt
    Eigen::Matrix3d relative_R;
    Eigen::Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        LOG(INFO) << "Not enough features or parallax; Move device around";
        return false;
    }

    // construct sfm vision init
    GlobalSFM sfm;
    sfm.setConfigfile(init_config_file_); 
    Eigen::Quaterniond Q[frameCount + 1];
    Eigen::Vector3d T[frameCount + 1];
    std::map<long, Eigen::Vector3d> sfmTrackedPoints;
    if (!sfm.construct(frameCount + 1, Q, T, l, relative_R, relative_T, sfmFeatures, sfmTrackedPoints))
    {
        LOG(WARNING) << "global SFM failed!";
        marginalizationFlag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    std::map<double, ImageFrame>::iterator frame_it;
    std::map<long, Eigen::Vector3d>::iterator it;
    frame_it = allImageFrame.begin();
    for (int i = 0; frame_it != allImageFrame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * Ric[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }

        if ((frame_it->first) > Headers[i])
        {
            i++;
        }

        Eigen::Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Eigen::Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        std::vector<cv::Point3f> pts_3_vector;
        std::vector<cv::Point2f> pts_2_vector;
        for (auto &idPts : frame_it->second.points)
        {
            long featureID = idPts.first;
            it = sfmTrackedPoints.find(featureID);
            if (it != sfmTrackedPoints.end())
            {
                Eigen::Vector3d world_pts = it->second;
                cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                pts_3_vector.push_back(pts_3);
                Eigen::Vector2d img_pts = idPts.second.p3D.head(2);
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
            }
        }

        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            LOG(INFO) << "pts_3_vector size " << pts_3_vector.size();
            LOG(INFO) << "Not enough points for solve pnp !";
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            LOG(INFO) << "solve pnp fail!";
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * Ric[0].transpose();
        frame_it->second.T = T_pnp;
    }

    if (monoVisualInitialAlign())
        return true;
    else
    {
        LOG(WARNING) << "misalign visual structure with IMU";
        return false;
    }
}

// stereo init
bool Estimator::stereoInitialization()
{
#ifdef TEST_PERF
    static int init_cnt = 0, sfm_cnt = 0, vi_cnt = 0;
    static int init_succ = 0, sfm_succ = 0, vi_succ = 0;
    init_cnt++;
#endif

    // check imu observibility
    {
        std::map<double, ImageFrame>::iterator frame_it;
        Eigen::Vector3d sum_g;
        for (frame_it = allImageFrame.begin(), frame_it++; frame_it != allImageFrame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }

        double var = 0;
        Eigen::Vector3d aver_g = sum_g * 1.0 / ((int)allImageFrame.size() - 1);
        for (frame_it = allImageFrame.begin(), frame_it++; frame_it != allImageFrame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = std::sqrt(var / ((int)allImageFrame.size() - 1));

        if (var < 0.25)
        {
            LOG(INFO) << "IMU excitation not enouth!";
            // return false;
        }
    }

#ifdef TEST_PERF
    utility::TicToc sfm_dt;
    sfm_cnt++;
#endif

    // global sfm
    std::vector<StereoSFMFeature> sfmFeatures;
    for (auto &idFeature : featureManager.pointFeatures)
    {
        auto &featureID = idFeature.first;
        auto &feature = idFeature.second;

        StereoSFMFeature sfmFeature;
        sfmFeature.state = false;
        sfmFeature.id = featureID;
        int imu_j = feature.startFrame;
        // insert observation
        for (auto &featureFrame : feature.featureFrames)
        {
            // mono observation
            Eigen::Vector2d pts_j = featureFrame.point.head(2);
            sfmFeature.observation.emplace_back(imu_j, pts_j);

            // stereo observation
            if (featureFrame.stereo)
            {
                Eigen::Vector2d ptsRight_j = featureFrame.pointRight.head(2);
                sfmFeature.observationRight.emplace_back(imu_j, ptsRight_j);
            }
            imu_j++;
        }
        sfmFeatures.emplace_back(sfmFeature);
    }

    // construct sfm vision init
    GlobalSFM sfm;
    sfm.setConfigfile(init_config_file_);
    Eigen::Quaterniond Q[frameCount + 1];
    Eigen::Vector3d T[frameCount + 1];
    std::map<long, Eigen::Vector3d> sfmTrackedPoints;
    if (!sfm.stereoConstruct(frameCount + 1, Q, T, Rl2r, tl2r, sfmFeatures, sfmTrackedPoints))
    {
        LOG(WARNING) << "global SFM failed!";
        marginalizationFlag = MARGIN_OLD;
        return false;
    }

#ifdef TEST_PERF
    sfm_succ++;
    vi_cnt++;
    {
        static double sum_dt = 0;
        static int cnt = 0;
        sum_dt += sfm_dt.toc();
        cnt++;
        printf("estimator sfm cost: %f\n", sum_dt / cnt);
    }
#endif

    // solve pnp for all frame
    std::map<double, ImageFrame>::iterator frame_it;
    std::map<long, Eigen::Vector3d>::iterator it;
    frame_it = allImageFrame.begin();
    for (int i = 0; frame_it != allImageFrame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * Ric[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }

        if ((frame_it->first) > Headers[i])
        {
            i++;
        }

        Eigen::Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Eigen::Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        std::vector<cv::Point3f> pts_3_vector;
        std::vector<cv::Point2f> pts_2_vector;
        for (auto &idPts : frame_it->second.points)
        {
            long featureID = idPts.first;
            it = sfmTrackedPoints.find(featureID);
            if (it != sfmTrackedPoints.end())
            {
                Eigen::Vector3d world_pts = it->second;
                cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                pts_3_vector.push_back(pts_3);
                Eigen::Vector2d img_pts = idPts.second.p3D.head(2);
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
            }
        }

        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            LOG(INFO) << "pts_3_vector size " << pts_3_vector.size();
            LOG(INFO) << "Not enough points for solve pnp !";
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            LOG(INFO) << "solve pnp fail!";
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * Ric[0].transpose();
        frame_it->second.T = T_pnp;
    }

    if (stereoVisualInitialAlign())
    {
#ifdef TEST_PERF
        {
            init_succ++;
            vi_succ++;
            printf("estimator sfm acc cost: %f\n", 1.f * sfm_succ / sfm_cnt);
            printf("estimator vi acc cost: %f\n", 1.f * vi_succ / vi_cnt);
            printf("estimator init acc cost: %f\n", 1.f * init_succ / init_cnt);
        }
#endif

        return true;
    }
    else
    {
        LOG(WARNING) << "misalign visual structure with IMU";
        return false;
    }
}

bool Estimator::monoVisualInitialAlign()
{
    utility::TicToc dt;

    // solve scale
    Eigen::VectorXd x;
    if (!initialAligment.visualIMUAlignment(allImageFrame, Bgs, g, x))
    {
        LOG(WARNING) << "solve g failed!";
        return false;
    }

    // change state
    for (int i = 0; i <= frameCount; i++)
    {
        allImageFrame[Headers[i]].is_key_frame = true;
        Eigen::Matrix3d R = allImageFrame[Headers[i]].R;
        Eigen::Vector3d t = allImageFrame[Headers[i]].T;
        Rs[i] = R;
        Ps[i] = t;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= windowSize; i++)
        pre_integrations[i]->repropagate(Eigen::Vector3d::Zero(), Bgs[i]);
    for (int i = frameCount; i >= 0; i--)
    {
        Ps[i] = s * Ps[i] - Rs[i] * tic[0] - (s * Ps[0] - Rs[0] * tic[0]);
        // LOG(INFO) << "Ps: " << Ps[i].transpose();
    }

    int kv = -1;
    std::map<double, ImageFrame>::iterator frame_i;
    for (frame_i = allImageFrame.begin(); frame_i != allImageFrame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    // calc g and init R0
    Eigen::Matrix3d R0 = utility::g2R(g);
    double yaw = utility::R2ypr(R0 * Rs[0]).x();
    R0 = utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;

    // recover roll and pitch in gravity system
    for (int i = 0; i <= frameCount; i++)
    {
        Ps[i] = R0 * Ps[i];
        Rs[i] = R0 * Rs[i];
        Vs[i] = R0 * Vs[i];
    }

    // triangulate feature in the slideWindow
    featureManager.clearDepth();
    featureManager.triangulate(frameCount, Ps, Rs, Ric, tic);

#ifdef TEST_PERF
    {
        static double sum_dt = 0;
        static int cnt = 0;
        sum_dt += dt.toc();
        cnt++;
        printf("estimator vi align cost: %f\n", sum_dt / cnt);
    }
#endif

    return true;
}

bool Estimator::stereoVisualInitialAlign()
{
    utility::TicToc dt;

    // solve scale
    Eigen::VectorXd x;
    if (!initialAligment.stereoVisualIMUAlignment(allImageFrame, Bgs, g, x))
    {
        LOG(WARNING) << "solve g failed!";
        return false;
    }

    // change state
    for (int i = 0; i <= frameCount; i++)
    {
        allImageFrame[Headers[i]].is_key_frame = true;
        Eigen::Matrix3d R = allImageFrame[Headers[i]].R;
        Eigen::Vector3d t = allImageFrame[Headers[i]].T;
        Rs[i] = R;
        Ps[i] = t;
    }

    for (int i = 0; i <= windowSize; i++)
        pre_integrations[i]->repropagate(Eigen::Vector3d::Zero(), Bgs[i]);
    for (int i = frameCount; i >= 0; i--)
    {
        Ps[i] = Ps[i] - Rs[i] * tic[0] - (Ps[0] - Rs[0] * tic[0]);
        // LOG(INFO) << "Ps: " << Ps[i].transpose();
    }

    int kv = -1;
    std::map<double, ImageFrame>::iterator frame_i;
    for (frame_i = allImageFrame.begin(); frame_i != allImageFrame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    // calc g and init R0
    Eigen::Matrix3d R0 = utility::g2R(g);
    double yaw = utility::R2ypr(R0 * Rs[0]).x();
    R0 = utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;

    // recover roll and pitch in gravity system
    for (int i = 0; i <= frameCount; i++)
    {
        Ps[i] = R0 * Ps[i];
        Rs[i] = R0 * Rs[i];
        Vs[i] = R0 * Vs[i];
    }

    // triangulate feature in the slideWindow
    featureManager.clearDepth();
    featureManager.triangulate(frameCount, Ps, Rs, Ric, tic);

#ifdef TEST_PERF
    {
        static double sum_dt = 0;
        static int cnt = 0;
        sum_dt += dt.toc();
        cnt++;
        printf("estimator vi align cost: %f\n", sum_dt / cnt);
    }
#endif

    return true;
}

bool Estimator::relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < windowSize; i++)
    {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
        corres = featureManager.getCorresponding(i, windowSize);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 && motionEstimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                // LOG(INFO)<<("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::originVector2Double()
{
    for (int i = 0; i <= windowSize; i++)
    {
        paraPose[i][0] = Ps[i].x();
        paraPose[i][1] = Ps[i].y();
        paraPose[i][2] = Ps[i].z();
        Eigen::Quaterniond q{Rs[i]};
        paraPose[i][3] = q.x();
        paraPose[i][4] = q.y();
        paraPose[i][5] = q.z();
        paraPose[i][6] = q.w();

        paraSpeedBias[i][0] = Vs[i].x();
        paraSpeedBias[i][1] = Vs[i].y();
        paraSpeedBias[i][2] = Vs[i].z();

        paraSpeedBias[i][3] = Bas[i].x();
        paraSpeedBias[i][4] = Bas[i].y();
        paraSpeedBias[i][5] = Bas[i].z();

        paraSpeedBias[i][6] = Bgs[i].x();
        paraSpeedBias[i][7] = Bgs[i].y();
        paraSpeedBias[i][8] = Bgs[i].z();
    }

    for (int i = 0; i < cameraNum; i++)
    {
        paraExPose[i][0] = tic[i].x();
        paraExPose[i][1] = tic[i].y();
        paraExPose[i][2] = tic[i].z();
        Eigen::Quaterniond q{Ric[i]};
        paraExPose[i][3] = q.x();
        paraExPose[i][4] = q.y();
        paraExPose[i][5] = q.z();
        paraExPose[i][6] = q.w();
    }

    // point feature
    Eigen::VectorXd dep = featureManager.getDepthVector();
    for (int i = 0; i < featureManager.getPointCount(); i++)
        paraFeature[i][0] = dep(i);

    paraTd[0][0] = td;
}

void Estimator::originDouble2Vector()
{
    Eigen::Vector3d originR0 = utility::R2ypr(Rs[0]);
    Eigen::Vector3d originP0 = Ps[0];

    if (failureOccur)
    {
        originR0 = utility::R2ypr(lastR0);
        originP0 = lastP0;
        failureOccur = 0;
    }

    Eigen::Vector3d optR0 = utility::R2ypr(Eigen::Quaterniond(paraPose[0][6], paraPose[0][3], paraPose[0][4], paraPose[0][5]).toRotationMatrix());
    double y_diff = originR0.x() - optR0.x();
    Eigen::Matrix3d rotDiff = utility::ypr2R(Eigen::Vector3d(y_diff, 0, 0));

    // TODO
    if (abs(abs(originR0.y()) - 90) < 1.0 || abs(abs(optR0.y()) - 90) < 1.0)
    {
        // LOG(INFO)<<("euler singular point!");
        rotDiff = Rs[0] * Eigen::Quaterniond(paraPose[0][6], paraPose[0][3], paraPose[0][4], paraPose[0][5]).toRotationMatrix().transpose();
    }

    // pose
    for (int i = 0; i <= windowSize; i++)
    {

        Rs[i] = rotDiff * Eigen::Quaterniond(paraPose[i][6], paraPose[i][3], paraPose[i][4], paraPose[i][5]).normalized().toRotationMatrix();

        Ps[i] =
            rotDiff * Eigen::Vector3d(paraPose[i][0] - paraPose[0][0], paraPose[i][1] - paraPose[0][1], paraPose[i][2] - paraPose[0][2]) + originP0;

        Vs[i] = rotDiff * Eigen::Vector3d(paraSpeedBias[i][0], paraSpeedBias[i][1], paraSpeedBias[i][2]);

        Bas[i] = Eigen::Vector3d(paraSpeedBias[i][3], paraSpeedBias[i][4], paraSpeedBias[i][5]);

        Bgs[i] = Eigen::Vector3d(paraSpeedBias[i][6], paraSpeedBias[i][7], paraSpeedBias[i][8]);
    }

    // extrinsics
    for (int i = 0; i < cameraNum; i++)
    {
        tic[i] = Eigen::Vector3d(paraExPose[i][0], paraExPose[i][1], paraExPose[i][2]);
        Ric[i] = Eigen::Quaterniond(paraExPose[i][6], paraExPose[i][3], paraExPose[i][4], paraExPose[i][5]).normalized().toRotationMatrix();
    }

    // point feature
    Eigen::VectorXd depVec = featureManager.getDepthVector();
    for (int i = 0; i < featureManager.getPointCount(); i++)
        depVec(i) = paraFeature[i][0];
    featureManager.setDepth(depVec);

    // td
    td = paraTd[0][0];
}

bool Estimator::failureDetection()
{
    return false;
    if (featureManager.lastTrackNum < 2)
    {
        LOG(WARNING) << "little feature " << featureManager.lastTrackNum;
        return true;
    }
    if (Bas[windowSize].norm() > 2.5)
    {
        LOG(WARNING) << "big IMU acc bias estimation %f", Bas[windowSize].norm();
        return true;
    }
    if (Bgs[windowSize].norm() > 1.0)
    {
        LOG(WARNING) << "big IMU gyr bias estimation %f", Bgs[windowSize].norm();
        return true;
    }

    Eigen::Vector3d tmp_P = Ps[windowSize];
    if ((tmp_P - lastP).norm() > 5)
    {
        // LOG(WARNING)<<" big translation";
        // return true;
    }
    if (std::abs(tmp_P.z() - lastP.z()) > 1)
    {
        // LOG(WARNING) << " big z translation";
        // return true;
    }
    Eigen::Matrix3d tmp_R = Rs[windowSize];
    Eigen::Matrix3d delta_R = tmp_R.transpose() * lastR;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        LOG(WARNING) << "big delta_angle ";
    }
    return false;
}

void Estimator::optimization()
{
    std::cout << "opt " << std::endl;
    {
        std::ofstream fout("/home/cl/workspace/project/intel/intel_stereo/src/intel_stereo/output/frameAndPoint.txt", std::ios::app);
        fout << " , frameCnt : " << Frame::frameIDCnt << " , pointCnt : " << Frame::pointIDCnt << std::endl;
        fout.close();
    }
    double tsolve_cost, tmargin_cost, twhole_cost;
    utility::TicToc tWhole, tPrepare;
    int cnt_visual0, cnt_visual1, cnt_visual2;
    int cnt_point;
    double pre_cost;

    originVector2Double();

    BaSolver::BA_problem problem;

    problem.initialStructure(config_file_);
    vector<shared_ptr<BaSolver::Pose>> vertexCams_vec;
    vector<shared_ptr<BaSolver::Motion>> vertexVB_vec;

    BaSolver::LossFunction *lossfunction;
    lossfunction = new BaSolver::CauchyLoss(1.0);

    //添加外参
    shared_ptr<BaSolver::Pose> vertexExt(new BaSolver::Pose());
    {
        vertexExt->SetParameters(paraExPose[0]);

        if (!estimateExtrinsic)
        {
            //ROS_DEBUG("fix extinsic param");
            // TODO:: set Hessian prior to zero
            vertexExt->SetFixed();
        }
        else{
            //ROS_DEBUG("estimate extinsic param");
        }

        problem.addExtParameterBlock(0, vertexExt);
    }
    //添加多相机双目外参
    shared_ptr<BaSolver::Pose> vertexRExt(new BaSolver::Pose());
    {
        if(stereo)
        {
            vertexRExt->SetParameters(paraExPose[1]);

            if(!estimateExtrinsic)
            {
                vertexRExt->SetFixed();
            }

            problem.addExtParameterBlock(1, vertexRExt);
        }
    }
    //添加图像位姿参数
    for (int i = 0; i < frameCount +1; i++)
    {
        shared_ptr<BaSolver::Pose> vertexCam(new BaSolver::Pose()); 
        vertexCam->SetParameters(paraPose[i]);
        vertexCams_vec.push_back(vertexCam);
        problem.addPoseParameterBlock(i, vertexCam);
    }
    // // 添加IMU参数
    for(int i = 0; i < frameCount + 1; i ++)
    {
        shared_ptr<BaSolver::Motion> vertexVB(new BaSolver::Motion());
        vertexVB->SetParameters(paraSpeedBias[i]);
        vertexVB_vec.push_back(vertexVB);
        problem.addIMUParameterBlock(i, vertexVB);
    }
    {
        // //添加先验约束
        if(ba_last_marginalization_info)
        {
            int num_residual = ba_last_marginalization_info->linearized_residuals.size();
            int num_vertices = ba_last_marginalization_info->keep_block_data.size();
            vector<string> typeInfo(num_vertices);
            shared_ptr<BaSolver::EdgePrior> priorEdge(new BaSolver::EdgePrior(ba_last_marginalization_info, num_residual, num_vertices, typeInfo));
            std::vector<std::shared_ptr<BaSolver::Vertex>> edge_vertex;
            int cnt = 0;
            //添加相机位姿参数
            for(int j = 0; j < frameCount + 1; j ++)
            {
                int sz = ba_last_marginalization_parameter_blocks.size();
                for(int i = 0; i < sz; i ++)
                {
                    if(ba_last_marginalization_parameter_blocks[i] == paraPose[j])
                    {
                        edge_vertex.push_back(vertexCams_vec[j]);
                        cnt ++;
                        break;
                    }
                }
            }
            //添加IMU参数
            for(int j = 0; j < frameCount + 1; j ++)
            {
                int sz = ba_last_marginalization_parameter_blocks.size();
                for(int i = 0; i < sz; i ++)
                {
                    if(ba_last_marginalization_parameter_blocks[i] == paraSpeedBias[j])
                    {
                        edge_vertex.push_back(vertexVB_vec[j]);
                        cnt ++;
                        break;
                    }
                }
            }
            priorEdge->SetVertex(edge_vertex);
            problem.addPriorResidualBlock(priorEdge);
        }
    }

    // // IMU约束
    for (int i = 0; i < frameCount; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        problem.addIMUResidualBlock(pre_integrations[j], vertexCams_vec[i], vertexVB_vec[i], vertexCams_vec[j], vertexVB_vec[j]);
    }
    
    // Visual Factor
    vector<shared_ptr<BaSolver::FeatureMeasure>> vertexPt_vec;
    {
        cnt_visual0 = 0; cnt_visual1 = 0; cnt_visual2 = 0;

        int f_m_cnt = 0;
        int feature_index = -1;
        // 遍历每一个特征
        for (auto &idFeature : featureManager.pointFeatures)
        {
            auto &feature = idFeature.second;
            feature.updateCnt();

            if (feature.usedNum < 4)
                continue;

            ++feature_index;

            int imu_i = feature.startFrame, imu_j = imu_i - 1;
            Eigen::Vector3d pts_i = feature.featureFrames[0].point;

            shared_ptr<BaSolver::FeatureMeasure> vertexPoint(new BaSolver::FeatureMeasure());
            int feature_id = feature.featureID;

            vertexPoint->SetParameters(paraFeature[feature_index]);
            problem.addFeatureParameterBlock(feature_id, vertexPoint);
            vertexPt_vec.push_back(vertexPoint);

            // 遍历所有的观测
            for (auto &it_per_frame : feature.featureFrames)
            {
                imu_j++;
    
                // if(f_m_cnt > 1) break;
                Vector3d pts_j = it_per_frame.point;
                if(imu_i != imu_j)
                {
                    problem.addFeatureResidualBlock(pts_i, pts_j, vertexCams_vec[imu_i], vertexCams_vec[imu_j], vertexPoint);
                    f_m_cnt ++;
                    cnt_visual0 ++;
                }

                if(stereo && it_per_frame.stereo)
                {
                    Eigen::Vector3d pts_j_right = it_per_frame.pointRight;

                    if(imu_i != imu_j)
                    {
                        problem.addStereoFeatureTwoFtwoCResidual(pts_i, pts_j_right, vertexCams_vec[imu_i], vertexCams_vec[imu_j], vertexPoint);
                        f_m_cnt ++;
                        cnt_visual1 ++;
                    }
                    else
                    {
                        problem.addStereoFeatureOneFtwoCResidual(pts_i, pts_j_right, vertexPoint);
                        f_m_cnt ++;
                        cnt_visual2 ++;
                    }  
                }

                
            }
        }
        cnt_point = feature_index + 1;   
    }
    pre_cost = tPrepare.toc();
    utility::TicToc tsolve;
    problem.solve();
    tsolve_cost = tsolve.toc();
    LOG_EVERY_N(INFO, logFreq) << "basolver solve cost: " << tsolve.toc() << "ms";
    

    originDouble2Vector();
    // printf("frameCount: %d \n", frameCount);

    if (frameCount < windowSize)
        return;

    utility::TicToc t_whole_marginalization;

     //边缘化
    {
        problem.preResidualJacobian();
        BaSolver::MarginalizationInfo *marginalization_info = new BaSolver::MarginalizationInfo();
        originVector2Double();
        if(marginalizationFlag == MARGIN_OLD)
        {
            //添加边缘化
            if(ba_last_marginalization_info)
            {
                vector<int> drop_set; drop_set.clear();
                int num_residual = ba_last_marginalization_info->linearized_residuals.size();
                int num_vertices = ba_last_marginalization_info->keep_block_data.size();
                vector<string> typeInfo(num_vertices);
                shared_ptr<BaSolver::EdgePrior> priorEdge(new BaSolver::EdgePrior(ba_last_marginalization_info, num_residual, num_vertices, typeInfo));
                vector<shared_ptr<BaSolver::Vertex>> edge_vertex; edge_vertex.clear();
                int cnt = 0;
                //添加相机位姿参数
                for(int j = 0; j < frameCount + 1; j ++)
                {
                    int sz = ba_last_marginalization_parameter_blocks.size();
                    for(int i = 0; i < sz; i ++)
                    {
                        if(ba_last_marginalization_parameter_blocks[i] == paraPose[j])
                        {
                            edge_vertex.push_back(vertexCams_vec[j]);
                            if(j == 0)
                            {
                                drop_set.push_back(cnt);
                            }
                            cnt ++;
                            break;
                        }
                    }
                }
                //添加IMU参数
                for(int j = 0; j < frameCount + 1; j ++)
                {
                    int sz = ba_last_marginalization_parameter_blocks.size();
                    for(int i = 0; i < sz; i ++)
                    {
                        if(ba_last_marginalization_parameter_blocks[i] == paraSpeedBias[j])
                        {
                            edge_vertex.push_back(vertexVB_vec[j]);
                            if(j == 0)
                            {
                                drop_set.push_back(cnt);
                            }
                            cnt ++;
                            break;
                        }
                    }
                }
                priorEdge->SetVertex(edge_vertex);
                BaSolver::ResidualBlockInfo *residual_block_info = new BaSolver::ResidualBlockInfo(priorEdge, drop_set);
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            {
            // IMU
                if(pre_integrations[1]->sum_dt < 10.0)
                {
                    std::shared_ptr<BaSolver::CostIMUFunction> imuEdge(new BaSolver::CostIMUFunction(pre_integrations[1]));
                    std::vector<std::shared_ptr<BaSolver::Vertex>> edge_vertex;
                    edge_vertex.push_back(vertexCams_vec[0]);
                    edge_vertex.push_back(vertexCams_vec[1]);
                    edge_vertex.push_back(vertexVB_vec[0]);
                    edge_vertex.push_back(vertexVB_vec[1]);
                    imuEdge->SetVertex(edge_vertex);
                    BaSolver::ResidualBlockInfo *residual_block_info = new BaSolver::ResidualBlockInfo(imuEdge, vector<int>{0, 2});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
            {
                //视觉
                int feature_index = -1;
                // 遍历每一个特征
                int f_m_cnt = 0;
                for (auto &idFeature : featureManager.pointFeatures)
                {
                    auto &feature = idFeature.second;
                    feature.updateCnt();

                    if (feature.usedNum < 4)
                        continue;

                    ++feature_index;

                    int imu_i = feature.startFrame, imu_j = imu_i - 1;
                    if (imu_i != 0)
                        continue;

                    Vector3d pts_i = feature.featureFrames[0].point;

                    // 遍历所有的观测
                    for (auto &it_per_frame : feature.featureFrames)
                    {
                        imu_j++;
                   
                        Vector3d pts_j = it_per_frame.point;
                        if(imu_i != imu_j)
                        {
                            std::shared_ptr<BaSolver::CostFunction> edge(new BaSolver::CostFunction(pts_i, pts_j));
                            std::vector<std::shared_ptr<BaSolver::Vertex>> edge_vertex;
                            edge_vertex.push_back(vertexCams_vec[imu_i]);
                            edge_vertex.push_back(vertexCams_vec[imu_j]);
                            edge_vertex.push_back(vertexPt_vec[feature_index]);

                            edge->SetVertex(edge_vertex);
                            edge->SetInformation(project_sqrt_info_);

                            edge->SetLossFunction(lossfunction);
                            // BaSolver::ResidualBlockInfo *residual_block_info = new BaSolver::ResidualBlockInfo(edge, vector<int>{0, 2});
                            BaSolver::ResidualBlockInfo *residual_block_info = new BaSolver::ResidualBlockInfo(edge, vector<int>{0, 2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }

                        if(stereo && it_per_frame.stereo)
                        {
                            Eigen::Vector3d pts_j_right = it_per_frame.pointRight;
                            if(imu_j != imu_i)
                            {
                                std::shared_ptr<BaSolver::CostTwoFrameTwoCamFunction> two_edge(new BaSolver::CostTwoFrameTwoCamFunction(pts_i, pts_j_right));
                                std::vector<std::shared_ptr<BaSolver::Vertex>> twoedge_vertex;
                                twoedge_vertex.push_back(vertexCams_vec[imu_i]);
                                twoedge_vertex.push_back(vertexCams_vec[imu_j]);
                     
                                twoedge_vertex.push_back(vertexPt_vec[feature_index]);

                                two_edge->SetVertex(twoedge_vertex);
                                two_edge->SetInformation(project_sqrt_info_);
                                two_edge->SetLossFunction(lossfunction);

                                BaSolver::ResidualBlockInfo *dou_residual_block_info = new BaSolver::ResidualBlockInfo(two_edge, vector<int>{0, 2});
                                marginalization_info->addResidualBlockInfo(dou_residual_block_info);
                            }
                            else
                            {
                                std::shared_ptr<BaSolver::CostOneFrameTwoCamFunction> one_edge(new BaSolver::CostOneFrameTwoCamFunction(pts_i, pts_j_right));

                                std::vector<std::shared_ptr<BaSolver::Vertex>> twoedge_vertex;
                    
                                twoedge_vertex.push_back(vertexPt_vec[feature_index]);

                                one_edge->SetVertex(twoedge_vertex);
                                one_edge->SetInformation(project_sqrt_info_);
                                one_edge->SetLossFunction(lossfunction);

                                BaSolver::ResidualBlockInfo *dou_residual_block_info = new BaSolver::ResidualBlockInfo(one_edge, vector<int>{0});
                                marginalization_info->addResidualBlockInfo(dou_residual_block_info);
                            }
                        }
                        f_m_cnt ++;
                    }
                }
            }
            marginalization_info->preMarginalize();
            marginalization_info->marginalize();
            std::unordered_map<int, double *> addr_shift;
            for (int i = 1; i <= frameCount; i++)
            {
                addr_shift[vertexCams_vec[i]->Id()] = paraPose[i - 1];
                addr_shift[vertexVB_vec[i]->Id()] = paraSpeedBias[i-1];
            }

            std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if(ba_last_marginalization_info) delete ba_last_marginalization_info;
            ba_last_marginalization_info = marginalization_info;
            ba_last_marginalization_parameter_blocks = parameter_blocks; 
        
            ba_last_marginalization_parameter_blocks = parameter_blocks;   
        }
        else 
        {
            if (ba_last_marginalization_info &&
            std::count(std::begin(ba_last_marginalization_parameter_blocks), std::end(ba_last_marginalization_parameter_blocks), paraPose[frameCount - 1]))
            {
                BaSolver::MarginalizationInfo *marginalization_info = new BaSolver::MarginalizationInfo();
                if(ba_last_marginalization_info)
                {
                    std::vector<int> drop_set; drop_set.clear();
                    std::vector<std::shared_ptr<BaSolver::Vertex>> edge_vertex;

                    int cnt = 0;
                    //添加相机位姿参数
                    for(int j = 0; j < frameCount + 1; j ++)
                    {
                        int sz = ba_last_marginalization_parameter_blocks.size();
                        for(int i = 0; i < sz; i ++)
                        {
                            if(ba_last_marginalization_parameter_blocks[i] == paraPose[j])
                            {
                                edge_vertex.push_back(vertexCams_vec[j]);
                                if(j == frameCount - 1)
                                {
                                    drop_set.push_back(cnt);
                                }
                                cnt ++;
                                break;
                            }
                        }
                    }
                    //添加IMU参数
                    for(int j = 0; j < frameCount+ 1; j ++)
                    {
                        int sz = ba_last_marginalization_parameter_blocks.size();
                        for(int i = 0; i < sz; i ++)
                        {
                            if(ba_last_marginalization_parameter_blocks[i] == paraSpeedBias[j])
                            {
                                edge_vertex.push_back(vertexVB_vec[j]);
                                cnt ++;
                                break;
                            }
                        }
                    }
                    int num_residual = ba_last_marginalization_info->linearized_residuals.size();
                    int num_vertices = ba_last_marginalization_info->keep_block_data.size();
                    std::vector<std::string> typeInfo(num_vertices);
                    std::shared_ptr<BaSolver::EdgePrior> priorEdge(new BaSolver::EdgePrior(ba_last_marginalization_info, num_residual, num_vertices, typeInfo));
               
                    priorEdge->SetVertex(edge_vertex);
                    BaSolver::ResidualBlockInfo *residual_block_info = new BaSolver::ResidualBlockInfo(priorEdge, drop_set);
                    // BaSolver::ResidualBlockInfo *residual_block_into = new BaSolver::ResidualBlockInfo(priorEdge, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_block_info);

                    marginalization_info->preMarginalize();
                    marginalization_info->marginalize();
                    std::unordered_map<int, double *> addr_shift;
                    for (int i = 0; i <= frameCount; i++)
                    {
                        if (i == frameCount - 1)
                            continue;
                        else if (i == frameCount)
                        {
                            addr_shift[vertexCams_vec[i]->Id()] = paraPose[i - 1];
                            addr_shift[vertexVB_vec[i]->Id()] = paraSpeedBias[i - 1];
                        }
                        else
                        {
                            addr_shift[vertexCams_vec[i]->Id()] = paraPose[i];
                            addr_shift[vertexVB_vec[i]->Id()] = paraSpeedBias[i];
                        }
                    }

                    vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
                    if (ba_last_marginalization_info)
                        delete ba_last_marginalization_info;
                    ba_last_marginalization_info = marginalization_info;
               
                    ba_last_marginalization_parameter_blocks = parameter_blocks;
                }
            }
        }
    }
    tmargin_cost = t_whole_marginalization.toc();
    twhole_cost = tWhole.toc();
    // {
    //     ofstream fout("/home/cl/workspace/project/intel/intel_stereo/src/vins_fusion/output/time/opt.txt", std::ios::app);

    //     fout.setf(ios::fixed, ios::floatfield); 
    //     fout.precision(5); 
    //     fout << Headers[windowSize - 2] << " " << cnt_point << " " << cnt_visual0 << " " << cnt_visual1 << " " << cnt_visual2 << " " << tsolve_cost << " " << tmargin_cost << " " << twhole_cost << std::endl;
    //     fout.close();

    // }
#ifdef TEST_PERF
    {
        static double sum_dt = 0;
        static double pre_dt = 0;
        static double hessian_dt = 0;
        static double linear_dt = 0;
        static double res_dt = 0;
        static double margin_dt = 0;
        static double solve_dt = 0;

        static int cnt = 0;
        sum_dt += twhole_cost;
        pre_dt += pre_cost;
        hessian_dt += problem.t_hessian_cost;
        linear_dt += problem.t_linear_solve_cost;
        res_dt += problem.t_res_cost;
        margin_dt += tmargin_cost;  
        solve_dt += tsolve_cost;

        cnt++;
        printf("optimization: pre : %f, hessian : %f, linear : %f, res :  %f, solve : %f, marg : %f, window : %f\n", 
                pre_dt / cnt, hessian_dt / cnt, linear_dt / cnt, res_dt / cnt, solve_dt / cnt,  margin_dt / cnt, sum_dt / cnt);
    }
#endif
    LOG_EVERY_N(INFO, logFreq) << "whole marginalization " << (marginalizationFlag == MarginalizationFlag::MARGIN_OLD ? "OLD costs: " : "SECOD_NEW costs: ") << t_whole_marginalization.toc() << "ms";
    LOG_EVERY_N(INFO, logFreq) << "whole optimization cost: " << tWhole.toc() << "ms";
}

void Estimator::slideWindow()
{
    utility::TicToc t_margin;
    if (marginalizationFlag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        backR0 = Rs[0];
        backP0 = Ps[0];
        if (frameCount == windowSize)
        {
            for (int i = 0; i < windowSize; i++)
            {
                Headers[i] = Headers[i + 1];
                Frames[i] = Frames[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linearAccBuf[i].swap(linearAccBuf[i + 1]);
                angularVelBuf[i].swap(angularVelBuf[i + 1]);

                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[windowSize] = Headers[windowSize - 1];
            Frames[windowSize] = Frames[windowSize - 1];
            Ps[windowSize] = Ps[windowSize - 1];
            Rs[windowSize] = Rs[windowSize - 1];

            Vs[windowSize] = Vs[windowSize - 1];
            Bas[windowSize] = Bas[windowSize - 1];
            Bgs[windowSize] = Bgs[windowSize - 1];

            // pre_integrations[windowSize] = std::make_shared<IntegrationBase>(acc_0, gyr_0, Bas[windowSize], Bgs[windowSize]);
            delete pre_integrations[windowSize];
            pre_integrations[windowSize] = new BaSolver::IntegrationBase{acc_0, gyr_0, Bas[windowSize], Bgs[windowSize]};

            dt_buf[windowSize].clear();
            linearAccBuf[windowSize].clear();
            angularVelBuf[windowSize].clear();

            if (true || solverFlag == INITIAL)
            {
                std::map<double, ImageFrame>::iterator it_0;
                it_0 = allImageFrame.find(t_0);
                allImageFrame.erase(allImageFrame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frameCount == windowSize)
        {
            Headers[frameCount - 1] = Headers[frameCount];
            Frames[frameCount - 1] = Frames[frameCount];
            Ps[frameCount - 1] = Ps[frameCount];
            Rs[frameCount - 1] = Rs[frameCount];

            for (unsigned int i = 0; i < dt_buf[frameCount].size(); i++)
            {
                double tmp_dt = dt_buf[frameCount][i];
                Eigen::Vector3d tmp_linear_acceleration = linearAccBuf[frameCount][i];
                Eigen::Vector3d tmp_angular_velocity = angularVelBuf[frameCount][i];

                pre_integrations[frameCount - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frameCount - 1].push_back(tmp_dt);
                linearAccBuf[frameCount - 1].push_back(tmp_linear_acceleration);
                angularVelBuf[frameCount - 1].push_back(tmp_angular_velocity);
            }

            Vs[frameCount - 1] = Vs[frameCount];
            Bas[frameCount - 1] = Bas[frameCount];
            Bgs[frameCount - 1] = Bgs[frameCount];

            delete pre_integrations[windowSize];
            pre_integrations[windowSize] = new BaSolver::IntegrationBase{acc_0, gyr_0, Bas[windowSize], Bgs[windowSize]};

            dt_buf[windowSize].clear();
            linearAccBuf[windowSize].clear();
            angularVelBuf[windowSize].clear();

            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    featureManager.removeFront(frameCount);
}

void Estimator::slideWindowOld()
{
    bool shift_depth = solverFlag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Eigen::Matrix3d R0, R1;
        Eigen::Vector3d P0, P1;
        R0 = backR0 * Ric[0];
        R1 = Rs[0] * Ric[0];
        P0 = backP0 + backR0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        featureManager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        featureManager.removeBack();
}

double Estimator::reprojectionError(Eigen::Matrix3d &Ri, Eigen::Vector3d &Pi, Eigen::Matrix3d &rici, Eigen::Vector3d &tici, Eigen::Matrix3d &Rj,
                                    Eigen::Vector3d &Pj, Eigen::Matrix3d &ricj, Eigen::Vector3d &ticj, double depth, Eigen::Vector3d &uvi,
                                    Eigen::Vector3d &uvj)
{
    Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head(2) - uvj.head(2);

    return residual.norm();
}

void Estimator::removeOutliers(const int pointsNum)
{
    // remove outliers after slide window optimization
    std::set<long> removeIndex;
    double totalErr = outliersRejection(removeIndex);
    // if (totalErr < outlierReprojectTh / 2)
    // featureManager.removeOutlier(removeIndex);

    // if (featureNum > 60 && removeIndex.size() < 20 && totalErr < 10)
    // else
    //     LOG(WARNING) << "too little feature or wrong outliers";

#ifdef TEST_PERF
    if (totalErr < 8)
    {
        static double error = 0;
        static int cnt = 0;
        error += totalErr;
        cnt++;
        printf("avg error: %f\n", totalErr);
        printf("point error: %f\n", 1.f * error / cnt);
    }
#endif

    LOG_EVERY_N(WARNING, logFreq) << "point size: " << pointsNum << " outlier size: " << removeIndex.size() << std::setprecision(4)
                                  << " point error: " << totalErr;
}

double Estimator::outliersRejection(std::set<long> &removeIndex)
{
    double totalErr = 0;
    int totalCnt = 0;
    for (auto &idFeature : featureManager.pointFeatures)
    {
        auto &featureID = idFeature.first;
        auto &feature = idFeature.second;

        feature.updateCnt();
        if (feature.usedNum < 4)
            continue;

        double err = 0;
        int errCnt = 0;

        int imu_i = feature.startFrame, imu_j = imu_i - 1;
        Eigen::Vector3d pts_i = feature.featureFrames[0].point;
        double depth = feature.estimatedDepth;
        for (auto &it_per_frame : feature.featureFrames)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Eigen::Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], Ric[0], tic[0], Rs[imu_j], Ps[imu_j], Ric[0], tic[0], depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                // printf("tmp_error %f\n", focalLength / 1.5 * tmp_error);
            }

            // need to rewrite projecton factor.........
            if (stereo && it_per_frame.stereo)
            {
                Eigen::Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    double tmp_error =
                        reprojectionError(Rs[imu_i], Ps[imu_i], Ric[0], tic[0], Rs[imu_j], Ps[imu_j], Ric[1], tic[1], depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", focalLength / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error =
                        reprojectionError(Rs[imu_i], Ps[imu_i], Ric[0], tic[0], Rs[imu_j], Ps[imu_j], Ric[1], tic[1], depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", focalLength / 1.5 * tmp_error);
                }
            }
        }

        double ave_err = err / errCnt;

        if (ave_err * focalLength > outlierReprojectTh)
            removeIndex.insert(featureID);

        totalErr += ave_err;
        totalCnt++;
    }

    return totalErr * focalLength / totalCnt;
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linearAcceleration, Eigen::Vector3d angularVelocity)
{
    std::unique_lock<std::mutex> lock(mPropagate);

    double dt = t - latestTime;
    latestTime = t;
    Eigen::Vector3d un_acc_0 = latestQ * (latestAcc_0 - latestBa) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latestGyr_0 + angularVelocity) - latestBg;
    latestQ = latestQ * utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latestQ * (linearAcceleration - latestBa) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latestP = latestP + dt * latestV + 0.5 * dt * dt * un_acc;
    latestV = latestV + dt * un_acc;
    latestAcc_0 = linearAcceleration;
    latestGyr_0 = angularVelocity;
}

void Estimator::updateLatestStates()
{
    // keyframe state for publish
    updateTime = Headers[frameCount];
    updateP = Ps[frameCount];
    updateQ = Rs[frameCount];
    updateV = Vs[frameCount];

    latestTime = Headers[frameCount] + td;
    latestP = Ps[frameCount];
    latestQ = Rs[frameCount];
    latestV = Vs[frameCount];
    latestBa = Bas[frameCount];
    latestBg = Bgs[frameCount];
    latestAcc_0 = acc_0;
    latestGyr_0 = gyr_0;

    std::queue<std::pair<double, Eigen::Vector3d>> tmp_accBuf, tmp_gyrBuf;
    {
        std::unique_lock<std::mutex> lock(mBuf);
        tmp_accBuf = accBuf;
        tmp_gyrBuf = gyrBuf;
    }

    while (!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
}
}  // namespace core
