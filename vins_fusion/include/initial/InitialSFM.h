/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include "utils/Marco.h"
#include "utils/Utility.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include "../../basolver/basolver.h"
/**
 * @brief data structure sfm feature
 * @refer VINS-Mono/Fusion SFMFuature
 */
struct SFMFeature
{
    /// sfm feature triangule flag
    bool state;
    /// sfm feature id
    long id;
    /// sfm feature 2D uv obs
    std::vector<std::pair<int, Eigen::Vector2d>> observation;
    /// sfm feature triangulate 3D postion
    double position[3];
    /// sfm feature depth
    double depth;
};
/**
 * @brief data structure sfm stereo feature
 * @refer VINS-Mono/Fusion SFMFuature
 */
struct StereoSFMFeature
{
    /// sfm feature triangule flag
    bool state;
    /// sfm feature id
    long id;
    /// sfm feature 2D uv obs
    std::vector<std::pair<int, Eigen::Vector2d>> observation;
    /// sfm feature 2D uv obs on right camera
    std::vector<std::pair<int, Eigen::Vector2d>> observationRight;
    /// sfm feature triangulate 3D postion
    double position[3];
    /// sfm feature depth
    double depth;
};

struct ReprojectionError
{
    ReprojectionError(const Eigen::Vector2d pt2D) : pt2D(pt2D) {}

    template <typename T>
    bool operator()(const T *const camera_R, const T *const camera_T, const T *point, T *residuals) const
    {
        // transform pt3D to camera frame
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0];
        p[1] += camera_T[1];
        p[2] += camera_T[2];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T f = (T)460;
        residuals[0] = f * (xp - (T)pt2D.x());
        residuals[1] = f * (yp - (T)pt2D.y());
        std::ofstream fout("/home/cl/workspace/project/intel/intel_stereo/src/vins_fusion/output/residual2.txt", std::ios::app);
        fout << residuals[0] << " " << residuals[1] << std::endl;
        fout.close();
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d pt2D)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(new ReprojectionError(pt2D)));
    }

    Eigen::Vector2d pt2D;
};

struct StereoReprojectionError
{
    StereoReprojectionError(const Eigen::Vector2d pt2D, const Eigen::Quaterniond Q_l2r, const Eigen::Vector3d t_l2r)
        : pt2D(pt2D), Ql2r(Q_l2r), tl2r(t_l2r)
    {
    }

    template <typename T>
    bool operator()(const T *const camera_R, const T *const camera_T, const T *point, T *residuals) const
    {
        // transform pt3D to left camera frame
        T ptLeft[3];
        ceres::QuaternionRotatePoint(camera_R, point, ptLeft);
        ptLeft[0] += camera_T[0];
        ptLeft[1] += camera_T[1];
        ptLeft[2] += camera_T[2];

        // transform pt3D from left to right
        T R[4], ptRight[3];
        R[0] = (T)Ql2r.w();
        R[1] = (T)Ql2r.x();
        R[2] = (T)Ql2r.y();
        R[3] = (T)Ql2r.z();
        ceres::QuaternionRotatePoint(R, ptLeft, ptRight);
        ptRight[0] += (T)tl2r.x();
        ptRight[1] += (T)tl2r.y();
        ptRight[2] += (T)tl2r.z();

        T xp = ptRight[0] / ptRight[2];
        T yp = ptRight[1] / ptRight[2];

        T f = (T)460;
        residuals[0] = f * (xp - (T)pt2D.x());
        residuals[1] = f * (yp - (T)pt2D.y());

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d pt2D, Eigen::Quaterniond Q_l2r, Eigen::Vector3d t_l2r)
    {
        return (new ceres::AutoDiffCostFunction<StereoReprojectionError, 2, 4, 3, 3>(new StereoReprojectionError(pt2D, Q_l2r, t_l2r)));
    }

    Eigen::Vector2d pt2D;
    Eigen::Quaterniond Ql2r;
    Eigen::Vector3d tl2r;
};

class GlobalSFM
{
  public:
    GlobalSFM();
    /**
     * @brief SFM Mono Initialization
     * @refer VINS-Mono/Fusion GlobalSFM
     * @param frame_num sliding window frame num
     * @param q sliding window frame rot
     * @param T sliding window frame pos
     * @param l sliding window major frame index
     * @param relative_R sliding window Rotation between major frame and last frame
     * @param relative_T sliding window Postion between major frame and last frame
     * @param sfm_f sfm features
     * @param sfm_tracked_points triangulate sfm feature(3D postion)
     * @return true SFM initial sucess
     * @return false SFM initial failed
     */
    bool construct(int frame_num, Eigen::Quaterniond q[], Eigen::Vector3d T[], int l, const Eigen::Matrix3d relative_R,
                   const Eigen::Vector3d relative_T, std::vector<SFMFeature> &sfm_f, std::map<long, Eigen::Vector3d> &sfm_tracked_points);
    /**
     * @brief SFM Stereo Initialization
     * @refer VINS-Mono/Fusion GlobalSFM
     * @param frame_num sliding window frame num
     * @param q sliding window frame rot
     * @param T sliding window frame pos
     * @param Rl2r Stereo extrinsic Rotation
     * @param tl2r Stereo extrinsic Postion
     * @param sfm_f sfm features
     * @param sfm_tracked_points triangulate sfm feature(3D postion)
     * @return true SFM initial sucess
     * @return false SFM initial failed
     */
    bool stereoConstruct(int frame_num, Eigen::Quaterniond q[], Eigen::Vector3d T[], const Eigen::Matrix3d &Rl2r, const Eigen::Vector3d &tl2r,
                         std::vector<StereoSFMFeature> &sfm_f, std::map<long, Eigen::Vector3d> &sfm_tracked_points);
    void setConfigfile(std::string config_file_)
    {
        config_file = config_file_;
    }
  private:
    /**
     * @brief PNP solve frame Pose
     * @refer VINS-Mono/Fusion GlobalSFM
     * @param R_initial PNP Rotation initial value
     * @param P_initial PNP Postion initial value
     * @param i PNP solve frame index
     * @param sfm_f sfm features
     * @return true solve PNP sucess
     * @return false olve PNP fail
     */
    bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, std::vector<SFMFeature> &sfm_f);
    /**
     * @brief PNP solve Stereo frame Pose
     * @refer VINS-Mono/Fusion GlobalSFM
     * @param R_initial PNP Rotation initial value
     * @param P_initial PNP Postion initial value
     * @param i PNP solve frame index
     * @param sfm_f sfm features
     * @return true solve PNP sucess
     * @return false olve PNP fail
     */
    bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, std::vector<StereoSFMFeature> &sfm_f);
    /**
     * @brief triangulate single Point
     * @refer VINS-Mono/Fusion GlobalSFM
     * @param Pose0 left frame Pose
     * @param Pose1 right frame Pose
     * @param point0 left frame point observation
     * @param point1 right frame point observation
     * @param point_3d triangulate 3D Point
     */
    void triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1, const Eigen::Vector2d &point0,
                          const Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    /**
     * @brief triangulate all points on the target frame
     * @refer VINS-Mono/Fusion GlobalSFM
     * @param frame0 left frame index
     * @param Pose0 left frame Pose
     * @param frame1 right frame index
     * @param Pose1 right frame Pose
     * @param sfm_f sfm feature
     */
    void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                              std::vector<SFMFeature> &sfm_f);
    /**
     * @brief triangulate all points on the target frame
     * @refer VINS-Mono/Fusion GlobalSFM
     * @param index target frame index
     * @param Pose target frame pose
     * @param R_l2r Stereo extrinsic Rotation
     * @param t_l2r Stereo extrinsic Position
     * @param sfmFeatures sfm features
     */
    void triangulateStereoFrame(int index, Eigen::Matrix<double, 3, 4> Pose[], const Eigen::Matrix3d &R_l2r, const Eigen::Vector3d &t_l2r,
                                std::vector<StereoSFMFeature> &sfmFeatures);

    int featureNum;
    std::string config_file;
};