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
#include "utils/Params.h"
#include "core/Frame.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

#include <list>
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <numeric>

class PointFeatureFrame
{
  public:
    PointFeatureFrame(const Eigen::Matrix<double, 7, 1> &ptUvVelocity, double td)
    {
        point.x() = ptUvVelocity(0);
        point.y() = ptUvVelocity(1);
        point.z() = ptUvVelocity(2);
        uv.x() = ptUvVelocity(3);
        uv.y() = ptUvVelocity(4);
        velocity.x() = ptUvVelocity(5);
        velocity.y() = ptUvVelocity(6);
        this->td = td;
        stereo = false;
    }

    PointFeatureFrame(const PointInfo &pointInfo, double td)
    {
        // Mono info
        point = pointInfo.p3D;
        uv = pointInfo.p2D;
        velocity = pointInfo.velocity.head(2);
        this->td = td;

        // Stereo info
        if (pointInfo.stereo)
        {
            pointRight = pointInfo.p3DRight;
            uvRight = pointInfo.p2DRight;
            velocityRight = pointInfo.velocityRight.head(2);
            stereo = true;
        }
        else
            stereo = false;
    }

    void rightObservation(const Eigen::Matrix<double, 7, 1> &ptUvVelocityRight)
    {
        pointRight.x() = ptUvVelocityRight(0);
        pointRight.y() = ptUvVelocityRight(1);
        pointRight.z() = ptUvVelocityRight(2);
        uvRight.x() = ptUvVelocityRight(3);
        uvRight.y() = ptUvVelocityRight(4);
        velocityRight.x() = ptUvVelocityRight(5);
        velocityRight.y() = ptUvVelocityRight(6);
        stereo = true;
    }

    double td;
    // Mono info
    Eigen::Vector3d point;
    Eigen::Vector2d uv;
    Eigen::Vector2d velocity;

    // Stereo info
    Eigen::Vector3d pointRight;
    Eigen::Vector2d uvRight;
    Eigen::Vector2d velocityRight;
    bool stereo;
};

class PointFeatureID
{
  public:
    const long featureID;
    int startFrame;
    std::vector<PointFeatureFrame> featureFrames;
    int usedNum;
    double estimatedDepth;
    int solveFlag;  // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    PointFeatureID(long featureID, int startFrame) : featureID(featureID), startFrame(startFrame), usedNum(0), estimatedDepth(-1.0), solveFlag(0) {}

    inline int endFrame()
    {
        return startFrame + featureFrames.size() - 1;
    }

    inline void updateCnt()
    {
        this->usedNum = featureFrames.size();
    }
};

namespace core
{
class FeatureManager
{
  public:
    FeatureManager();

    void clearState();

    // get feature info
    int getPointCount();

    void setDepth(const Eigen::VectorXd &x);

    Eigen::VectorXd getDepthVector();

    bool addFeatureCheckParallax(int frameCnt, const Frame &feature, double td);

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    void removeFailures();

    void clearDepth();

    void triangulate(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], const std::vector<Eigen::Matrix3d> &Ric,
                     const std::vector<Eigen::Vector3d> &tic);

    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);

    void removeBack();

    void removeFront(int frame_count);

    void removeOutlier(std::set<long> &outlierIndex);

    // data
    std::unordered_map<long, PointFeatureID> pointFeatures;

    int lastTrackNum;

  private:
    double compensatedParallax2(const PointFeatureID &it_per_id, int frame_count);

    // base params
    const bool stereo;
    const int windowSize;
    const double initDepth;
    const double focalLength;
    const double keyframeParallax;
};
}  // namespace core
