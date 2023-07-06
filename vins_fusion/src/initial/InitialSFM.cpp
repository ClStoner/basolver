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

#include "initial/InitialSFM.h"

GlobalSFM::GlobalSFM() {}

void GlobalSFM::triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1, const Eigen::Vector2d &point0,
                                 const Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool GlobalSFM::solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, std::vector<SFMFeature> &sfm_f)
{
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for (int j = 0; j < featureNum; j++)
    {
        if (sfm_f[j].state != true)
            continue;
        Eigen::Vector2d point2d;
        for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
        {
            if (sfm_f[j].observation[k].first == i)
            {
                Eigen::Vector2d img_pts = sfm_f[j].observation[k].second;
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
                cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
                pts_3_vector.push_back(pts_3);
                break;
            }
        }
    }

    if (int(pts_2_vector.size()) < 15)
    {
        printf("unstable features tracking, please slowly move you device!\n");
        if (int(pts_2_vector.size()) < 10)
            return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    if (!pnp_succ)
    {
        return false;
    }
    cv::Rodrigues(rvec, r);
    // cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;
}

bool GlobalSFM::solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, std::vector<StereoSFMFeature> &sfm_f)
{
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for (int j = 0; j < featureNum; j++)
    {
        if (sfm_f[j].state != true)
            continue;

        for (unsigned k = 0; k < sfm_f[j].observation.size(); k++)
        {
            if (sfm_f[j].observation[k].first == i)
            {
                Eigen::Vector2d img_pts = sfm_f[j].observation[k].second;
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
                cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
                pts_3_vector.push_back(pts_3);
                break;
            }
        }
    }

    if (int(pts_2_vector.size()) < 15)
    {
        printf("unstable features tracking, please slowly move you device!\n");
        if (int(pts_2_vector.size()) < 10)
            return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    if (!pnp_succ)
    {
        return false;
    }
    cv::Rodrigues(rvec, r);
    // cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;
}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                                     std::vector<SFMFeature> &sfm_f)
{
    CHECK_NE(frame0, frame1);

    for (int j = 0; j < featureNum; j++)
    {
        if (sfm_f[j].state == true)
            continue;

        bool has_0 = false, has_1 = false;
        Eigen::Vector2d point0;
        Eigen::Vector2d point1;
        for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
        {
            if (sfm_f[j].observation[k].first == frame0)
            {
                point0 = sfm_f[j].observation[k].second;
                has_0 = true;
            }
            if (sfm_f[j].observation[k].first == frame1)
            {
                point1 = sfm_f[j].observation[k].second;
                has_1 = true;
            }
        }

        if (has_0 && has_1)
        {
            Eigen::Vector3d point_3d;
            triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
            // cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
        }
    }
}

void GlobalSFM::triangulateStereoFrame(int index, Eigen::Matrix<double, 3, 4> Pose[], const Eigen::Matrix3d &R_l2r, const Eigen::Vector3d &t_l2r,
                                       std::vector<StereoSFMFeature> &sfmFeatures)
{
    for (auto &feature : sfmFeatures)
    {
        if (feature.state == true)
            continue;

        bool has0 = false, has1 = false, has2 = false;
        int idx0, idx1, idx2;
        Eigen::Vector2d point0, point1, point2;

        for (auto &featureFrame : feature.observation)
        {
            if (featureFrame.first < index)
            {
                has2 = true;
                idx2 = featureFrame.first;
                point2 = featureFrame.second;
            }
            else if (featureFrame.first == index)
            {
                has0 = true;
                idx0 = featureFrame.first;
                point0 = featureFrame.second;
            }
        }

        for (auto &featureFrame : feature.observationRight)
        {
            if (featureFrame.first == index)
            {
                has1 = true;
                idx1 = featureFrame.first;
                point1 = featureFrame.second;
            }
        }

        if (has0 && has1)  // stereo
        {
            Eigen::Matrix4d PoseLeft = Eigen::Matrix4d::Identity(), PoseRight = Eigen::Matrix4d::Identity(), Posel2r = Eigen::Matrix4d::Identity();
            PoseLeft.block(0, 0, 3, 4) = Pose[idx0];
            Posel2r.block(0, 0, 3, 3) = R_l2r;
            Posel2r.block(0, 3, 3, 1) = t_l2r;
            PoseRight = Posel2r * PoseLeft;

            Eigen::Vector3d p3D;
            triangulatePoint(PoseLeft.block(0, 0, 3, 4), PoseRight.block(0, 0, 3, 4), point0, point1, p3D);
            feature.state = true;
            feature.position[0] = p3D(0);
            feature.position[1] = p3D(1);
            feature.position[2] = p3D(2);
        }
        else if (has0 && has2)  // mono
        {
            Eigen::Vector3d p3D;
            triangulatePoint(Pose[idx0], Pose[idx2], point0, point2, p3D);
            feature.state = true;
            feature.position[0] = p3D(0);
            feature.position[1] = p3D(1);
            feature.position[2] = p3D(2);
        }
    }

    int cnt = 0;
    for (auto &feature : sfmFeatures)
    {
        if (feature.state == true)
            cnt++;
    }
    // LOG(INFO) << "cnt: " << cnt << " total: " << featureNum;
}

// 	q w_R_cam t w_R_cam
//  c_rotation cam_R_w
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frameNum, Eigen::Quaterniond q[], Eigen::Vector3d T[], int l, const Eigen::Matrix3d relative_R,
                          const Eigen::Vector3d relative_T, std::vector<SFMFeature> &sfm_f, std::map<long, Eigen::Vector3d> &sfm_tracked_points)
{
    featureNum = sfm_f.size();
    // cout << "set 0 and " << l << " as known " << endl;
    // have relative_r relative_t
    // intial two view
    q[l].w() = 1;
    q[l].x() = 0;
    q[l].y() = 0;
    q[l].z() = 0;
    T[l].setZero();
    q[frameNum - 1] = q[l] * Eigen::Quaterniond(relative_R);
    T[frameNum - 1] = relative_T;


    // rotation world frame to cam frame
    Eigen::Matrix3d c_Rotation[frameNum];
    Eigen::Vector3d c_Translation[frameNum];
    Eigen::Quaterniond c_Quat[frameNum];
    Eigen::Matrix<double, 3, 4> Pose[frameNum];

    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    c_Quat[frameNum - 1] = q[frameNum - 1].inverse();
    c_Rotation[frameNum - 1] = c_Quat[frameNum - 1].toRotationMatrix();
    c_Translation[frameNum - 1] = -1 * (c_Rotation[frameNum - 1] * T[frameNum - 1]);
    Pose[frameNum - 1].block<3, 3>(0, 0) = c_Rotation[frameNum - 1];
    Pose[frameNum - 1].block<3, 1>(0, 3) = c_Translation[frameNum - 1];

    // 1: trangulate between l ----- frameNum - 1
    // 2: solve pnp l + 1; trangulate l + 1 ------- frameNum - 1;
    for (int i = l; i < frameNum - 1; i++)
    {
        // solve pnp
        if (i > l)
        {
            Eigen::Matrix3d R_initial = c_Rotation[i - 1];
            Eigen::Vector3d P_initial = c_Translation[i - 1];
            if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
                return false;
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }

        // triangulate point based on the solve pnp result
        triangulateTwoFrames(i, Pose[i], frameNum - 1, Pose[frameNum - 1], sfm_f);
    }

    // 3: triangulate l-----l+1 l+2 ... frameNum -2
    for (int i = l + 1; i < frameNum - 1; i++)
        triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

    // 4: solve pnp l-1; triangulate l-1 ----- l
    //             l-2              l-2 ----- l
    for (int i = l - 1; i >= 0; i--)
    {
        // solve pnp
        Eigen::Matrix3d R_initial = c_Rotation[i + 1];
        Eigen::Vector3d P_initial = c_Translation[i + 1];
        if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        // triangulate
        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
    }

    // 5: triangulate all other points
    for (int j = 0; j < featureNum; j++)
    {
        if (sfm_f[j].state == true)
            continue;
        if ((int)sfm_f[j].observation.size() >= 2)
        {
            Eigen::Vector2d point0, point1;
            int frame_0 = sfm_f[j].observation[0].first;
            point0 = sfm_f[j].observation[0].second;
            int frame_1 = sfm_f[j].observation.back().first;
            point1 = sfm_f[j].observation.back().second;
            Eigen::Vector3d point_3d;
            triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
            // cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
        }
    }

    // //   // full BA
    // double c_rotation[frameNum][4];
    // double c_translation[frameNum][3];
    // ceres::Problem problem;
    // ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();
    // // cout << " begin full BA " << endl;
    // for (int i = 0; i < frameNum; i++)
    // {
    //     // double array for ceres
    //     c_translation[i][0] = c_Translation[i].x();
    //     c_translation[i][1] = c_Translation[i].y();
    //     c_translation[i][2] = c_Translation[i].z();
    //     c_rotation[i][0] = c_Quat[i].w();
    //     c_rotation[i][1] = c_Quat[i].x();
    //     c_rotation[i][2] = c_Quat[i].y();
    //     c_rotation[i][3] = c_Quat[i].z();
    //     problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
    //     problem.AddParameterBlock(c_translation[i], 3);
    //     // if (i == l)
    //     // {
    //     // }
    //     if (i == l || i == frameNum - 1)
    //     {
    //         problem.SetParameterBlockConstant(c_rotation[i]);
    //         problem.SetParameterBlockConstant(c_translation[i]);
    //     }
    // }
    // {
    //     std::ofstream fout("/home/cl/workspace/project/intel/intel_stereo/src/vins_fusion/output/opt_before2.txt");
    //     for(int i = 0; i < frameNum; i ++)
    //     {
    //         fout << c_translation[i][0] << " " << c_translation[i][1] << " " << c_translation[i][2] << " " << c_rotation[i][1] << " " <<  c_rotation[i][2] << " " << c_rotation[i][3] << " " << c_rotation[i][0] << std::endl;
    //     }
    //     fout.close();
    // }

    // for (int i = 0; i < featureNum; i++)
    // {
    //     if (sfm_f[i].state != true)
    //         continue;
    //     for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
    //     {
    //         int l = sfm_f[i].observation[j].first;
            // ceres::CostFunction *cost_function = ReprojectionError::Create(sfm_f[i].observation[j].second);

    //         problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], sfm_f[i].position);
    //     }
    // }

    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // // options.minimizer_progress_to_stdout = true;
    // options.max_solver_time_in_seconds = 0.2;
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);

    // if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    // {
    //     // cout << "vision only BA converge" << endl;
    // }
    // else
    // {
    //     // cout << "vision only BA not converge " << endl;
    //     return false;
    // }

    // {
    //     std::ofstream fout("/home/cl/workspace/project/intel/intel_stereo/src/vins_fusion/output/opt_after2.txt");
    //     for(int i = 0; i < frameNum; i ++)
    //     {
    //         fout << c_translation[i][0] << " " << c_translation[i][1] << " " << c_translation[i][2] << " " << c_rotation[i][1] << " " <<  c_rotation[i][2] << " " << c_rotation[i][3] << " " << c_rotation[i][0] << std::endl;
    //     }
    //     fout.close();
    // }
    // // save BA results
    // for (int i = 0; i < frameNum; i++)
    // {
    //     q[i].w() = c_rotation[i][0];
    //     q[i].x() = c_rotation[i][1];
    //     q[i].y() = c_rotation[i][2];
    //     q[i].z() = c_rotation[i][3];
    //     q[i] = q[i].inverse();
    //     // cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
    // }
    // for (int i = 0; i < frameNum; i++)
    // {

    //     T[i] = -1.0 * (q[i] * Eigen::Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    //     // cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
    // }
    // for (unsigned i = 0; i < sfm_f.size(); i++)
    // {
    //     if (sfm_f[i].state)
    //         sfm_tracked_points[sfm_f[i].id] = Eigen::Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
    // }


    BaSolver::BA_problem problem;
 
    problem.initialStructure(config_file);

    std::vector<std::shared_ptr<BaSolver::Pose>> vectorCams_vec;
	std::vector<std::shared_ptr<BaSolver::FeatureMeasureXYZ>> vectorFeatureXYZ_vec;
    // // full BA
    double c_rotation[frameNum][4];
    double c_translation[frameNum][3];
    double c_rt[frameNum][7];
    double ext[7];
    memset(ext, 0, sizeof(ext));
    ext[6] = 1.0;

    
    //添加外参，单位旋转、零平移
    std::shared_ptr<BaSolver::Pose> vertexExt(new BaSolver::Pose());
    {
        vertexExt->SetParameters(ext);
        vertexExt->SetFixed();
        problem.addExtParameterBlock(0, vertexExt);
    }
    
    for (int i = 0; i < frameNum; i++)
    {
        // double array for ceres
        Eigen::Quaterniond Qd = c_Quat[i].inverse();

        Eigen::Vector3d Vt = Qd * (-c_Translation[i]);
        c_rt[i][0] = Vt.x();
        c_rt[i][1] = Vt.y();
        c_rt[i][2] = Vt.z();
        c_rt[i][3] = Qd.x();
        c_rt[i][4] = Qd.y();
        c_rt[i][5] = Qd.z();
        c_rt[i][6] = Qd.w();
        std::shared_ptr<BaSolver::Pose> vertexCam(new BaSolver::Pose());
		vertexCam->SetParameters(c_rt[i]);
		if(i == l || i == frameNum - 1) 
        {
            vertexCam->SetFixed();
        }
		problem.addPoseParameterBlock(i, vertexCam); 
        vectorCams_vec.push_back(vertexCam);
    }
  

    for (int i = 0; i < featureNum; i++)
    {
        if (sfm_f[i].state != true)
            continue;
        std::shared_ptr<BaSolver::FeatureMeasureXYZ> featureXYZ(new BaSolver::FeatureMeasureXYZ());
        featureXYZ->SetParameters(sfm_f[i].position);
        problem.addFeatureXYZParameterBlock(featureXYZ);
        vectorFeatureXYZ_vec.push_back(featureXYZ);
        for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
        {
            int l = sfm_f[i].observation[j].first;
            Eigen::Vector3d pts(sfm_f[i].observation[j].second.x(), sfm_f[i].observation[j].second.y(), 1.0);
            problem.addFeatureXYZResidualBlock(pts, vectorCams_vec[l], featureXYZ);
        }
    }
    problem.solve();

  //  // save BA results
    for (int i = 0; i < frameNum; i++)
    {
        q[i].w() = c_rt[i][6];
        q[i].x() = c_rt[i][3];
        q[i].y() = c_rt[i][4];
        q[i].z() = c_rt[i][5];
    }
    for (int i = 0; i < frameNum; i++)
    {
        T[i] = Eigen::Vector3d(c_rt[i][0], c_rt[i][1], c_rt[i][2]);
    }
    for (unsigned i = 0; i < sfm_f.size(); i++)
    {
        if (sfm_f[i].state)
            sfm_tracked_points[sfm_f[i].id] = Eigen::Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
    }
    return true;
}

bool GlobalSFM::stereoConstruct(int frameNum, Eigen::Quaterniond Q[], Eigen::Vector3d t[], const Eigen::Matrix3d &Rl2r, const Eigen::Vector3d &tl2r,
                                std::vector<StereoSFMFeature> &sfmFeatures, std::map<long, Eigen::Vector3d> &sfmTrackedPoints)
{
    utility::TicToc td;

    featureNum = sfmFeatures.size();

    // init start frame
    Q[0] = Eigen::Quaterniond::Identity();
    t[0] = Eigen::Vector3d::Zero();

    // rotation world frame to cam frame
    Eigen::Matrix3d c_Rotation[frameNum];
    Eigen::Vector3d c_Translation[frameNum];
    Eigen::Quaterniond c_Quat[frameNum];
    Eigen::Matrix<double, 3, 4> Pose[frameNum];

    // init start frame
    c_Quat[0] = Q[0].inverse();
    c_Rotation[0] = c_Quat[0].toRotationMatrix();
    c_Translation[0] = -c_Rotation[0] * t[0];
    Pose[0].block(0, 0, 3, 3) = c_Rotation[0];
    Pose[0].block(0, 3, 3, 1) = c_Translation[0];

    // init sfm frames
    for (int i = 0; i < frameNum; i++)
    {
        if (i > 0)
        {
            Eigen::Matrix3d R_initial = c_Rotation[i - 1];
            Eigen::Vector3d P_initial = c_Translation[i - 1];
            if (!solveFrameByPnP(R_initial, P_initial, i, sfmFeatures))
                return false;
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }
        // LOG(WARNING) << std::fixed << std::setprecision(5) << "t " << i << " " << c_Translation[i](0) << " " << c_Translation[i](1) << " "
        //              << c_Translation[i](2) << " q: "
        //              << " " << utility::R2ypr(c_Rotation[i]).transpose();
        // triangulate point based on the solve pnp result
        triangulateStereoFrame(i, Pose, Rl2r, tl2r, sfmFeatures);
    }

    // // full BA
    // Eigen::Quaterniond Ql2r{Rl2r};
    // double c_rotation[frameNum][4];
    // double c_translation[frameNum][3];
    // ceres::Problem problem;
    // ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    // ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();
    // for (int i = 0; i < frameNum; i++)
    // {
    //     // double array for ceres
    //     c_translation[i][0] = c_Translation[i].x();
    //     c_translation[i][1] = c_Translation[i].y();
    //     c_translation[i][2] = c_Translation[i].z();
    //     c_rotation[i][0] = c_Quat[i].w();
    //     c_rotation[i][1] = c_Quat[i].x();
    //     c_rotation[i][2] = c_Quat[i].y();
    //     c_rotation[i][3] = c_Quat[i].z();
    //     problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
    //     problem.AddParameterBlock(c_translation[i], 3);
    // }
    // // set constant parameter block
    // problem.SetParameterBlockConstant(c_rotation[0]);
    // problem.SetParameterBlockConstant(c_translation[0]);
    // // problem.SetParameterBlockConstant(c_translation[frameNum - 1]);

    // int cnt = 0;
    // for (int i = 0; i < featureNum; i++)
    // {
    //     if (sfmFeatures[i].state == false)
    //         continue;

    //     // add feature parameter block
    //     problem.AddParameterBlock(sfmFeatures[i].position, 3);

    //     // mono factor
    //     for (unsigned j = 0; j < sfmFeatures[i].observation.size(); j++)
    //     {
    //         int idx = sfmFeatures[i].observation[j].first;
    //         ceres::CostFunction *cost_function = ReprojectionError::Create(sfmFeatures[i].observation[j].second);
    //         problem.AddResidualBlock(cost_function, loss_function, c_rotation[idx], c_translation[idx], sfmFeatures[i].position);
    //         cnt++;
    //     }

    //     // stereo factor
    //     for (unsigned j = 0; j < sfmFeatures[i].observationRight.size(); j++)
    //     {
    //         int idx = sfmFeatures[i].observationRight[j].first;
    //         ceres::CostFunction *cost_function = StereoReprojectionError::Create(sfmFeatures[i].observationRight[j].second, Ql2r, tl2r);
    //         problem.AddResidualBlock(cost_function, loss_function, c_rotation[idx], c_translation[idx], sfmFeatures[i].position);
    //         cnt++;
    //     }
    // }

    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // options.minimizer_progress_to_stdout = true;
    // // options.max_solver_time_in_seconds = 0.2;
    // options.max_num_iterations = 15;
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);
    // // LOG(INFO) << summary.FullReport();
    // LOG(INFO) << summary.BriefReport();

    // if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 1e4)
    // {
    //     // LOG(INFO) << "vision only BA converge";
    // }
    // else
    // {
    //     // LOG(INFO) << "vision only BA not converge ";
    //     return false;
    // }

    // // save BA results
    // for (int i = 0; i < frameNum; i++)
    // {
    //     Q[i] = Eigen::Quaterniond{c_rotation[i][0], c_rotation[i][1], c_rotation[i][2], c_rotation[i][3]}.inverse();
    //     t[i] = -1.0 * (Q[i] * Eigen::Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));

    //     LOG(WARNING) << i << std::fixed << std::setprecision(4) << "t: " << t[i](0) << " " << t[i](1) << " " << t[i](2) << " q: "
    //                  << " " << utility::R2ypr(Q[i].toRotationMatrix()).transpose();
    // }

    // for (unsigned i = 0; i < sfmFeatures.size(); i++)
    // {
    //     if (sfmFeatures[i].state)
    //         sfmTrackedPoints[sfmFeatures[i].id] = Eigen::Vector3d(sfmFeatures[i].position[0], sfmFeatures[i].position[1], sfmFeatures[i].position[2]);
    // }



    // Eigen::Quaterniond Ql2r{Rl2r};
    BaSolver::BA_problem problem;
    problem.initialStructure(config_file);

    std::vector<std::shared_ptr<BaSolver::Pose>> vectorCams_vec;
	std::vector<std::shared_ptr<BaSolver::FeatureMeasureXYZ>> vectorFeatureXYZ_vec;
    double c_rotation[frameNum][4];
    double c_translation[frameNum][3];
    double c_rt[frameNum][7];
    double ext[7];
    memset(ext, 0, sizeof(ext));
    ext[6] = 1.0;

    
    //添加外参，单位旋转、零平移
    std::shared_ptr<BaSolver::Pose> vertexExt(new BaSolver::Pose());
    {
        vertexExt->SetParameters(ext);
        vertexExt->SetFixed();
        problem.addExtParameterBlock(0, vertexExt);
    }
    for (int i = 0; i < frameNum; i++)
    {
        // double array for ceres
        Eigen::Quaterniond rwb = c_Quat[i].inverse();

        Eigen::Vector3d twb = rwb * (-c_Translation[i]);
        c_rt[i][0] = twb.x();
        c_rt[i][1] = twb.y();
        c_rt[i][2] = twb.z();
        c_rt[i][3] = rwb.x();
        c_rt[i][4] = rwb.y();
        c_rt[i][5] = rwb.z();
        c_rt[i][6] = rwb.w();
        std::shared_ptr<BaSolver::Pose> vertexCam(new BaSolver::Pose());
		vertexCam->SetParameters(c_rt[i]);
        if(i == 0)
        {
            vertexCam->SetFixed();
        }
		problem.addPoseParameterBlock(i, vertexCam); 
        vectorCams_vec.push_back(vertexCam);
    }
    int cnt = 0;
    for (int i = 0; i < featureNum; i++)
    {
        if (sfmFeatures[i].state == false)
            continue;


        std::shared_ptr<BaSolver::FeatureMeasureXYZ> featureXYZ(new BaSolver::FeatureMeasureXYZ());
        featureXYZ->SetParameters(sfmFeatures[i].position);
        problem.addFeatureXYZParameterBlock(featureXYZ);
        vectorFeatureXYZ_vec.push_back(featureXYZ);
        for (int j = 0; j < int(sfmFeatures[i].observation.size()); j++)
        {
            int l = sfmFeatures[i].observation[j].first;
            Eigen::Vector3d pts(sfmFeatures[i].observation[j].second.x(), sfmFeatures[i].observation[j].second.y(), 1.0);
            problem.addFeatureXYZResidualBlock(pts, vectorCams_vec[l], featureXYZ);
        }
    }
    problem.solve();

    // save BA results
    for (int i = 0; i < frameNum; i++)
    {
        Q[i].w() = c_rt[i][6];
        Q[i].x() = c_rt[i][3];
        Q[i].y() = c_rt[i][4];
        Q[i].z() = c_rt[i][5];
        t[i] = Eigen::Vector3d(c_rt[i][0], c_rt[i][1], c_rt[i][2]);

        LOG(WARNING) << i << std::fixed << std::setprecision(4) << "t: " << t[i](0) << " " << t[i](1) << " " << t[i](2) << " q: "
                     << " " << utility::R2ypr(Q[i].toRotationMatrix()).transpose();
    }

    for (unsigned i = 0; i < sfmFeatures.size(); i++)
    {
        if (sfmFeatures[i].state)
            sfmTrackedPoints[sfmFeatures[i].id] = Eigen::Vector3d(sfmFeatures[i].position[0], sfmFeatures[i].position[1], sfmFeatures[i].position[2]);
    }

    LOG(INFO) << "stereo sfm construct init cost: " << td.toc();

    return true;
}
