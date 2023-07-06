/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "core/FeatureManager.h"

namespace core
{
FeatureManager::FeatureManager()
    : stereo(params.Stereo || params.Depth), windowSize(params.windowSize), initDepth(params.initDepth), focalLength(params.focalLength),
      keyframeParallax(params.keyframeParallax)
{
    LOG(INFO) << "FeatureManager create";
}

void FeatureManager::clearState()
{
    pointFeatures.clear();
}

int FeatureManager::getPointCount()
{
    int cnt = 0;
    for (auto &it : pointFeatures)
    {
        auto &feature = it.second;
        feature.updateCnt();
        if (feature.usedNum >= 4)
            cnt++;
    }
    return cnt;
}

void FeatureManager::setDepth(const Eigen::VectorXd &depVec)
{
    int featureIndex = -1;
    for (auto &idFeature : pointFeatures)
    {
        auto &feature = idFeature.second;
        feature.updateCnt();
        if (feature.usedNum < 4)
            continue;

        feature.estimatedDepth = 1.0 / depVec(++featureIndex);
        // ROS_INFO("feature id %d , startFrame %d, depth %f ", it_per_id->feature_id, it_per_id-> startFrame, it_per_id->estimatedDepth);
        if (feature.estimatedDepth < 0)
            feature.solveFlag = 2;

        else
            feature.solveFlag = 1;
    }
}

Eigen::VectorXd FeatureManager::getDepthVector()
{
    Eigen::VectorXd depVec(getPointCount());
    int feature_index = -1;
    for (auto &idFeature : pointFeatures)
    {
        auto &feature = idFeature.second;
        feature.updateCnt();
        if (feature.usedNum < 4)
            continue;

        depVec(++feature_index) = 1.0 / feature.estimatedDepth;
    }

    return depVec;
}

bool FeatureManager::addFeatureCheckParallax(int frameCnt, const Frame &feature, double td)
{
    lastTrackNum = 0;
    int longTrackNum = 0, newFeatureNum = 0;

    // add point feature
    for (auto &point : feature.points)
    {
        PointFeatureFrame featureFrame(point, td);

        long featureID = point.id;
        auto iter = pointFeatures.find(featureID);
        if (iter == pointFeatures.end())
        {
            PointFeatureID feature(featureID, frameCnt);
            feature.featureFrames.emplace_back(featureFrame);
            pointFeatures.emplace(featureID, feature);
            newFeatureNum++;
        }
        else if (iter->second.featureID == featureID)
        {
            iter->second.featureFrames.emplace_back(featureFrame);
            lastTrackNum++;

            if (iter->second.featureFrames.size() >= 4)
                longTrackNum++;
        }
    }

    // judge keyframe by tracking state and pts disparity
    if (frameCnt < 2 || lastTrackNum < 20 || longTrackNum < 40 || newFeatureNum > 0.5 * lastTrackNum)
        return true;

    int parallaxNum = 0;
    double parallaxSum = 0;
    for (auto &idFeature : pointFeatures)
    {
        auto &feature = idFeature.second;
        if (feature.startFrame <= frameCnt - 2 && feature.startFrame + int(feature.featureFrames.size()) - 1 >= frameCnt - 1)
        {
            parallaxSum += compensatedParallax2(feature, frameCnt);
            parallaxNum++;
        }
    }

    if (parallaxNum != 0)
    {
        double lastAverageParallax = parallaxSum / parallaxNum;
        return lastAverageParallax >= keyframeParallax;
    }
    else
        return true;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
    for (auto &idFeature : pointFeatures)
    {
        auto &feature = idFeature.second;
        if (feature.startFrame <= frame_count_l && feature.endFrame() >= frame_count_r)
        {
            int idx_l = frame_count_l - feature.startFrame;
            int idx_r = frame_count_r - feature.startFrame;

            Eigen::Vector3d a = feature.featureFrames[idx_l].point;
            Eigen::Vector3d b = feature.featureFrames[idx_r].point;

            corres.emplace_back(a, b);
        }
    }

    return corres;
}

void FeatureManager::removeFailures()
{
    // remove depth < 0 point feature
    for (auto it = pointFeatures.begin(), it_next = pointFeatures.begin(); it != pointFeatures.end(); it = it_next)
    {
        it_next++;
        if (it->second.solveFlag == 2)
            pointFeatures.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &idFeature : pointFeatures)
        idFeature.second.estimatedDepth = -1.0;

    // for (auto &it_per_id : feature)
    //     it_per_id.estimatedDepth = -1;
}

void FeatureManager::triangulate(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], const std::vector<Eigen::Matrix3d> &Ric,
                                 const std::vector<Eigen::Vector3d> &tic)
{
    // Point feature triangulate
    for (auto &idFeature : pointFeatures)
    {
        auto &feature = idFeature.second;

        if (feature.estimatedDepth > 0)
            continue;

        if (stereo && feature.featureFrames[0].stereo)  // stereo triangulate
        {
            int imu_i = feature.startFrame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * Ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;

            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
            Eigen::Matrix3d R1 = Rs[imu_i] * Ric[1];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;

            Eigen::Vector3d point0 = feature.featureFrames[0].point;
            Eigen::Vector3d point1 = feature.featureFrames[0].pointRight;
            Eigen::Vector3d point3d = utility::triangulatePoint(R0, t0, point0, R1, t1, point1);

            double depth = point3d.z();
            if (depth > 0)
                feature.estimatedDepth = depth;
            else
                feature.estimatedDepth = initDepth;

            continue;
        }
        else if (feature.featureFrames.size() > 1)  // mono triangulate
        {
            int imu_i = feature.startFrame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * Ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;

            imu_i++;
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_i] * Ric[0];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;

            Eigen::Vector3d point0 = feature.featureFrames[0].point;
            Eigen::Vector3d point1 = feature.featureFrames[1].point;
            Eigen::Vector3d point3d = utility::triangulatePoint(R0, t0, point0, R1, t1, point1);

            double depth = point3d.z();
            if (depth > 0)
                feature.estimatedDepth = depth;
            else
                feature.estimatedDepth = initDepth;

            continue;
        }

        feature.updateCnt();
        if (feature.usedNum < 4)
            continue;
    }
}

void FeatureManager::removeOutlier(std::set<long> &outlierIndex)
{
    for (auto &id : outlierIndex)
    {
        auto iter = pointFeatures.find(id);
        if (iter != pointFeatures.end())
            pointFeatures.erase(iter);
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{

    // remove point feature and shift depth
    for (auto it = pointFeatures.begin(), it_next = pointFeatures.begin(); it != pointFeatures.end(); it = it_next)
    {
        it_next++;

        if (it->second.startFrame != 0)
            it->second.startFrame--;
        else
        {
            Eigen::Vector3d uv_i = it->second.featureFrames[0].point;
            it->second.featureFrames.erase(it->second.featureFrames.begin());
            if (it->second.featureFrames.size() < 2)
            {
                pointFeatures.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->second.estimatedDepth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->second.estimatedDepth = dep_j;
                else
                    it->second.estimatedDepth = initDepth;
            }
        }
    }
}

void FeatureManager::removeBack()
{
    // remove point feature
    for (auto it = pointFeatures.begin(), it_next = pointFeatures.begin(); it != pointFeatures.end(); it = it_next)
    {
        it_next++;

        if (it->second.startFrame != 0)
            it->second.startFrame--;
        else
        {
            it->second.featureFrames.erase(it->second.featureFrames.begin());
            if (it->second.featureFrames.size() == 0)
                pointFeatures.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    // remove point feature
    for (auto it = pointFeatures.begin(), it_next = pointFeatures.begin(); it != pointFeatures.end(); it = it_next)
    {
        it_next++;

        if (it->second.startFrame == frame_count)
        {
            it->second.startFrame--;
        }
        else
        {
            int j = windowSize - 1 - it->second.startFrame;
            if (it->second.endFrame() < frame_count - 1)
                continue;
            it->second.featureFrames.erase(it->second.featureFrames.begin() + j);
            if (it->second.featureFrames.empty())
                pointFeatures.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const PointFeatureID &it_per_id, int frame_count)
{
    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    const PointFeatureFrame &frame_i = it_per_id.featureFrames[frame_count - 2 - it_per_id.startFrame];
    const PointFeatureFrame &frame_j = it_per_id.featureFrames[frame_count - 1 - it_per_id.startFrame];

    double ans = 0;
    Eigen::Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Eigen::Vector3d p_i = frame_i.point;
    Eigen::Vector3d p_i_comp;

    // int r_i = frame_count - 2;
    // int r_j = frame_count - 1;
    // p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = std::max(ans, std::sqrt(std::min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
}  // namespace core
