/**
 * @file problem.h
 * @author ClStoner (ClStoner@163.com)
 * @brief 求解器类的实现，参考了手写VIO的代码
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_BAPROBLEM_H
#define BASOLVER_BAPROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>

#include "corrector.h"
#include "problem.h"
#include "edge.h"
#include "costIMUfunction.h"
#include "costFourPoseGraphfunction.h"
#include "costPoseGraphfunction.h"
#include "costfunction.h"
#include "costXYZfunction.h"
#include "costOneFrameTwoCamfunction.h"
#include "costTwoFrameTwoCamfunction.h"
#include "vertex.h"
#include "pose.h"
#include "pose_yaw.h"
#include "feature_xyz.h"
#include "feature_inverse_depth.h"
#include "motion.h"
#include "parameters.h"
#include "integration_base.h"
#include "eigen_types.h"
#include "loss_function.h"
#include "marginalization.h"
#include "mymatrix.h"
typedef unsigned long ulong;

namespace BaSolver{

typedef unsigned long ulong;
//    typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;
/// @brief 所有优化节点的集合




/**
 * @brief 求解器类的完整实现，主要内容包括了参数块的添加、约束块的添加、以及求解的实现
 */

class BA_problem {
public:
    /**
     * @brief 地图点类型，分逆深度表示与3D点表示
     * 
     */
    enum class PointType
    {
        POINT_INVERSE_DEPTH,    //逆深度表示
        POINT_XYZ               // 3D点表示
    };
    /**
     * @brief 求解问题的类型
     * 一般通用问题、传统SLAM问题、增量式SLAM问题
     * 
     */
    enum class ProblemType {
        SLAM_BAPROBLEM,           //SLAM BA问题求解求解问题
        SLAM_POSEGRAPH              //SLAM POSE GRAPH问题求解
        // SLAM_INIT,               //3D-pose 初始化优化问题
    };
    enum class LossFunctionType
    {
        Cauchy,
        Huber,
        Tukey
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @brief 构造函数
     * 
     */
    BA_problem();

    /**
     * @brief 析构函数
     * 
     */
    ~BA_problem();

    /**
     * @brief 配置求解器相关参数，包括求解器类型、损失和函数类型、求解器最大迭代次数等相关参数
     * 
     * @param config_file 参数文件
     */
    void initialStructure(std::string config_file);
    /**
     * @brief 提供外部添加外采模块接口
     * 
     * @param id 相机id
     * @param vertex 添加节点
     */
    void addExtParameterBlock(int id, std::shared_ptr<Vertex> vertex);
    /**
     * @brief 提供外部添加位姿参数模块接口
     * 
     * @param id 位姿在滑窗中id
     * 
     * @param vertex 添加的节点
     */
    void addPoseParameterBlock(int id, std::shared_ptr<Vertex> vertex);
    /**
     * @brief 提供外部添加IMU参数模块接口
     * 
     * @param id 位姿在滑窗中id
     * 
     * @param vertex 添加的节点
     */
    void addIMUParameterBlock(int id, std::shared_ptr<Vertex> vertex);
    /**
     * @brief 提供外部添加3D点逆深度参数模块接口
     * 
     * @param id IMU信息在滑窗中id
     * 
     * @param vertex 添加的节点
     */
    void addFeatureParameterBlock(int feature_id, std::shared_ptr<Vertex> vertex);
    /**
     * @brief 提供外部3D点XYZ形式参数模块接口
     * 
     * @param vertex 添加的节点
     */
    void addFeatureXYZParameterBlock(std::shared_ptr<Vertex> vertex);
    /**
     * @brief 提供外部添加IMU残差约束块接口
     * 
     * @param _pre_integration 预积分
     * @param pose_i 位姿 i 节点
     * @param bias_i bias i 节点
     * @param pose_j 位姿 j 节点
     * @param bias_j bias j 节点
     */
    void addIMUResidualBlock(IntegrationBase* _pre_integration, std::shared_ptr<Vertex> pose_i, std::shared_ptr<Vertex> bias_i, std::shared_ptr<Vertex> pose_j, std::shared_ptr<Vertex> bias_j);
    /**
     * @brief 提供外部添加视觉约束块接口
     * 
     * @param pts_i 3D点在位姿 i 中的归一化坐标
     * @param pts_j 3D点在位姿 j 中的归一化坐标
     * @param pose_i 位姿 i 节点
     * @param pose_j 位姿 j 节点
     * @param feature_point 3D点逆深度节点
     */
    void addFeatureResidualBlock(Eigen::Vector3d pts_i, Eigen::Vector3d pts_j, std::shared_ptr<Vertex> pose_i, std::shared_ptr<Vertex> pose_j, std::shared_ptr<Vertex> feature_point);
    /**
     * @brief 提供外部添加3D点与位姿的视觉约束块接口
     * 
     * @param pts 3D点在位姿中的归一化坐标
     * @param pose_i 位姿节点
     * @param feature_xyz_point 3D点XYZ节点
     */
    void addFeatureXYZResidualBlock(Eigen::Vector3d pts, std::shared_ptr<Vertex> pose, std::shared_ptr<FeatureMeasureXYZ> feature_xyz_point);
    /**
     * @brief 提供双目添加视觉约束块接口，表示路标点被同一帧的左右目观测到
     * 
     * @param pts_i 3D点在观测帧左目下的归一化坐标
     * @param pts_j 3D点在观测帧右目下的归一化坐标
     * @param feature_point 3D点逆深度节点
     */
    void addStereoFeatureOneFtwoCResidual(Eigen::Vector3d pts_i, Eigen::Vector3d pts_j, std::shared_ptr<Vertex> feature_point);
    /**
     * @brief 提供双目添加视觉约束块接口，表示路标点被主导帧的左目观测到、同时被当前帧右目观测到
     * 
     * @param pts_i 3D点在主导帧左目下的归一化坐标
     * @param pts_j 3D点在当前帧右目下的归一化坐标
     * @param pose_i 位姿 i 节点
     * @param pose_j 位姿 j 节点
     * @param feature_point 3D点逆深度节点
     */
    void addStereoFeatureTwoFtwoCResidual(Eigen::Vector3d pts_i, Eigen::Vector3d pts_j, std::shared_ptr<Vertex> pose_i, std::shared_ptr<Vertex> pose_j, std::shared_ptr<Vertex> feature_point);
    /**
     * @brief 提供外部添加先验约束块接口
     * 
     * @param priorEdge 先验约束
     */
    void addPriorResidualBlock(std::shared_ptr<Edge> priorEdge);
     /**
     * @brief 预处理视觉约束中（包括双目）残差雅克比计算对应变换矩阵
     * 
     */
    void preResidualJacobian();
    /**
     * @brief 提供外部优化接口
     * @return true 
     * @return false 
     */
    bool solve();
     /// @brief 预处理时间
    double pre_cost = 0;
    ///@brief 设置优化变量顺序所需时间
    double t_setordering_cost = 0.0;
    /// @brief 构建信息矩阵所需要的时间（包括 残差计算、雅克比计算、核函数矫正残差雅克比以及信息矩阵构建）
    double t_hessian_cost = 0.0;
    /// @brief 线性求解所需时间
    double t_linear_solve_cost = 0.0;
    /// @brief 更新优化变量所需时间
    double t_update_cost = 0.0;
    /// @brief 判断当前迭代是否合适所需时间
    double t_judge_cost = 0.0;
    /// @brief 统计残差计算耗时（包括rho的分子分母的计算）
    double t_res_cost = 0.0;
    /// @brief 整个求解器求解的时间
    double t_solve_cost = 0.0;
    /// @brief 是否在终端打印中间求解结果
    bool progress_to_stdout;
private:

    /**
     * @brief 向求解器添加参数块
     * 
     * @param vertex 
     * @return true 
     * @return false 
     */
    bool AddVertex(std::shared_ptr<Vertex> vertex);
    /**
     * @brief 向求解器添加约束快
     * 
     * @param edge 
     * @return true 
     * @return false 
     */
    bool AddEdge(std::shared_ptr<Edge> edge);


    /**
     * @brief 设置各顶点的ordering_index
     * 
     */
    void SetOrdering();
    
    /**
     * @brief 在SLAM问题中为新节点设置 ordering_index
     * 
     * @param v 添加的节点
     */
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    /**
     * @brief 构建信息矩阵
     * 
     */
    void MakeHessian();

    /**
     * @brief 计算残差理论值的变化量
     * 
     */
    void ComputeModelCostChange();
  
    /**
     * @brief 解线性方程
     * 
     */
    void SolveLinearSystem();

    /**
     * @brief 更新状态变量
     * 
     */
    void UpdateStates();
    /**
     * @brief 当 update 后残差会变大，需要退回去，重来
     * 
     */
    void RollbackStates(); 

    /**
     * @brief 判断一个节点是否为Pose节点
     * 
     * @param v 节点
     * @return true 
     * @return false 
     */
    bool IsPoseVertex(std::shared_ptr<Vertex> v);

    /**
     * @brief 判断一个节点是否为landmark节点
     * 
     * @param v 节点
     * @return true 
     * @return false 
     */
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

    /**
     * @brief 检查ordering是否正确
     * 
     * @return true 
     * @return false 
     */
    bool CheckOrdering();
    /**
     * @brief 初始化求解器相关变量
     * 
     */
    void InitProblem();

    /**
     * @brief 计算LM算法的初始Lambda
     * 
     */
    void ComputeLambdaInitLM();
    /**
     * @brief LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
     * 
     * @return true 
     * @return false 
     */
    bool IsGoodStepInLM();

   
    /// @brief Lambda
    double currentLambda_;
    /// @brief 上一次迭代的残差
    double currentChi_;
    /// @brief 控制 Lambda 缩放大小
    double ni_;
    /// @brief 地图点表示类型
    PointType pointType_;
    /// @brief 优化问题类型
    ProblemType problemType_;

    /// @brief 损失核函数类型
    LossFunction *lossfunction_;
    /// @brief 优化节点集合
    HashVertex verticies_;

    /// @brief 定义向量 diagonal_ = sqrt(diag(J^T*T))
    double *diagonal_;
    /// @brief 定义向量D_=diagonal*diagonal
    double *D_;
    /// @brief  信息矩阵对角元素最小值阈值
    double min_diagonal_;
    /// @brief 信息矩阵对角元素最大值阈值
    double max_diagonal_;
    /// @brief 优化变量模长
    double x_norm;
    /// @brief 当前优化变量变化模长
    double delta_norm;
    /// @brief 理论残差变化量
    double model_cost_change_; 
    /// @brief 当前残差
    double tempChi_;

    int *stRow;

    
    /// @brief 所有约束的集合
    HashEdge edges_;

    /// @brief 以ordering排序的pose顶点
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_; 
    /// @brief 以ordering排序的landmark顶点
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;

    /// @brief 是否是第一次迭代
    bool first_iter_ = true;
    /// @brief 求解器最大迭代次数
    int max_iteration_;
    /// @brief 求解器中外参的个数
    int cnt_ext_;
    /// @brief 求解器中位姿的个数
    int cnt_pose_;
    /// @brief 求解器中IMU信息的个数
    int cnt_motion_;
    /// @brief 求解器中地图点的个数
    int cnt_landmark_;
     
};

}


#endif
