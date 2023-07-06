#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include "utility/tic_toc.h"
#include "basolver.h"
#ifdef USE_OPENMP

#include <omp.h>

#endif

using namespace std;


// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}


namespace BaSolver {


BA_problem::BA_problem()  
{
    pointType_ = PointType::POINT_INVERSE_DEPTH;
    problemType_ = ProblemType::SLAM_BAPROBLEM;
    max_iteration_ = 50;
    progress_to_stdout = false;
    lossfunction_ = nullptr;

    pointToVertex_.clear();
    vertexToPoint_.clear();

    poseToVertex_.clear();
    vertexToPose_.clear();

    extToVertex_.clear();
    vertexToExt_.clear();

    startFrame_.clear();
    verticies_.clear();
    Ric.clear();
    Rci.clear();
    tic.clear();
    tci.clear();

    R_cj_ci.clear();
    t_cj_ci.clear();
    R_cj_w.clear();
    R_bj_ci.clear();
    t_bj_ci.clear();
    R_cj_bi.clear();
    R_rcj_lci.clear();
    t_rcj_lci.clear();
    R_rcj_w.clear();
    R_rcj_bi.clear();


    cnt_ext_ = 0 ;
    cnt_pose_ = 0;
    cnt_motion_ = 0;
    cnt_landmark_ = 0;
}

BA_problem::~BA_problem() 
{
    global_vertex_id = 0;
    global_edge_id = 0;
    delete lossfunction_;
    delete[] Hpp;
    delete[] Hpm;
    delete[] Hmp;
    delete[] Hmm;  
    delete[] Hmm_inv;
    delete[] tempH;
    delete[] b;
    delete[] diagonal_;
    delete[] D_;  
    delete[] delta_x;
    delete[] stRow;
}
void BA_problem::initialStructure(std::string config_file)
{   
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cout << "no path" << std::endl;
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }
    max_iteration_ = fsSettings["max_num_iterations"];
    int flag = fsSettings["progress_to_stdout"];

    if(flag) progress_to_stdout = true;
    else progress_to_stdout = false;
    int problemType = fsSettings["problem_type"];

    
    STEREO = false;
    {
        int flag = fsSettings["stereo"];
        if(flag) STEREO = true;
    }
    int lossfunctionType = fsSettings["lossfunction_type"];
    if(lossfunctionType == 0) lossfunction_ = nullptr;
    else 
    {
        double s = fsSettings["delta"];
        if(lossfunctionType == 1)
        {
            lossfunction_ = new CauchyLoss(s);
        }
        else if(lossfunctionType == 2)
        {
            lossfunction_ = new HuberLoss(s);
        }
        else {
            lossfunction_ = new TukeyLoss(s);
        }
    }
    int pointType = fsSettings["point_type"];

    if(pointType == 0) pointType_ = PointType::POINT_INVERSE_DEPTH;
    else pointType_ = PointType::POINT_XYZ;

    if(problemType == 0) problemType_ == ProblemType::SLAM_BAPROBLEM;
    else problemType_ == ProblemType::SLAM_POSEGRAPH;
    
    R_cj_ci.resize(WINDOW_SIZE + 1, std::vector<Eigen::Matrix3d>(WINDOW_SIZE + 1));
    t_cj_ci.resize(WINDOW_SIZE + 1, std::vector<Eigen::Vector3d>(WINDOW_SIZE + 1));

    R_cj_w.resize(WINDOW_SIZE + 1, std::vector<Eigen::Matrix3d>(WINDOW_SIZE + 1));

    R_bj_ci.resize(WINDOW_SIZE + 1, std::vector<Eigen::Matrix3d>(WINDOW_SIZE + 1));
    t_bj_ci.resize(WINDOW_SIZE + 1, std::vector<Eigen::Vector3d>(WINDOW_SIZE + 1));

    R_cj_bi.resize(WINDOW_SIZE + 1, std::vector<Eigen::Matrix3d>(WINDOW_SIZE + 1));

    if(STEREO)
    {
        R_rcj_lci.resize(WINDOW_SIZE + 1, std::vector<Eigen::Matrix3d>(WINDOW_SIZE + 1));
        t_rcj_lci.resize(WINDOW_SIZE + 1, std::vector<Eigen::Vector3d>(WINDOW_SIZE + 1));
        R_rcj_w.resize(WINDOW_SIZE + 1, std::vector<Eigen::Matrix3d>(WINDOW_SIZE + 1));
        R_rcj_bi.resize(WINDOW_SIZE + 1, std::vector<Eigen::Matrix3d>(WINDOW_SIZE + 1));
    }
}
void BA_problem::InitProblem()
{
    pre_cost = 0.0;
    t_setordering_cost = 0.0;
    t_hessian_cost = 0.0;
    t_linear_solve_cost = 0.0;
    t_update_cost = 0.0;
    t_judge_cost = 0.0;
    t_res_cost = 0;
    t_solve_cost = 0.0;
}
double squaredNorm(double *a, int sz)
{
    double res = 0;
    for(int i = 0; i < sz; i ++)
    {
        res += (a[i] * a[i]);
    }
    return res;
}
void BA_problem::addExtParameterBlock(int id, std::shared_ptr<Vertex> vertex)
{
    cnt_ext_ ++;
    extToVertex_[id] = vertex->Id();
    vertexToExt_[vertex->Id()] = id;

    double const *param_ext = vertex->Parameters();
    Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
    Vec3 t(param_ext[0], param_ext[1], param_ext[2]);
    Mat33 r = qic.normalized().toRotationMatrix();
    Mat33 rt = r.transpose();
    Vec3 tt = -rt * t;
    Ric.push_back(r);
    Rci.push_back(rt);
    tic.push_back(t);
    tci.push_back(tt);
    AddVertex(vertex);
}
void BA_problem::addPoseParameterBlock(int id, std::shared_ptr<Vertex> vertex)
{
    cnt_pose_ ++;
    poseToVertex_[id] = vertex->Id();
    vertexToPose_[vertex->Id()] = id;
    AddVertex(vertex);
}
void BA_problem::addIMUParameterBlock(int id, std::shared_ptr<Vertex> vertex)
{
    cnt_motion_ ++;
    AddVertex(vertex);
}
void BA_problem::addFeatureParameterBlock(int id, std::shared_ptr<Vertex> vertex)
{
    cnt_landmark_ ++;
    pointToVertex_[id] = vertex->Id();
    vertexToPoint_[vertex->Id()] = id;
    AddVertex(vertex);
}
void BA_problem::addFeatureXYZParameterBlock(std::shared_ptr<Vertex> vertex)
{
    AddVertex(vertex);
}
void BA_problem::addIMUResidualBlock(IntegrationBase* pre_integration, std::shared_ptr<Vertex> pose_i, std::shared_ptr<Vertex> bias_i, std::shared_ptr<Vertex> pose_j, std::shared_ptr<Vertex> bias_j)
{
    std::shared_ptr<CostIMUFunction> imuEdge(new CostIMUFunction(pre_integration));
    std::vector<std::shared_ptr<BaSolver::Vertex>> edge_vertex;
    edge_vertex.push_back(pose_i);
    edge_vertex.push_back(pose_j);
    edge_vertex.push_back(bias_i);
    edge_vertex.push_back(bias_j);
    imuEdge->SetVertex(edge_vertex);
    AddEdge(imuEdge);
}
void BA_problem::addFeatureResidualBlock(Eigen::Vector3d pts_i, Eigen::Vector3d pts_j, std::shared_ptr<Vertex> pose_i, std::shared_ptr<Vertex> pose_j, std::shared_ptr<Vertex> feature_point)
{   
    std::shared_ptr<CostFunction> edge(new CostFunction(pts_i, pts_j));
    std::vector<std::shared_ptr<BaSolver::Vertex>> edge_vertex;
    //添加视觉约束，节点的顺序为 pose_i, pose_j pose_Fea
    edge_vertex.push_back(pose_i);
    edge_vertex.push_back(pose_j);
    edge_vertex.push_back(feature_point);
    edge->SetVertex(edge_vertex);

    edge->SetInformation(project_sqrt_info_);
    edge->SetLossFunction(lossfunction_);

    if(startFrame_.find(feature_point->Id()) == startFrame_.end())
    {
        startFrame_[feature_point->Id()] = make_pair(vertexToPose_[pose_i->Id()], 2);
    }
    else 
    {
        startFrame_[feature_point->Id()].second ++;
    }

    AddEdge(edge);
}
void BA_problem::addFeatureXYZResidualBlock(Eigen::Vector3d pts, std::shared_ptr<Vertex> pose, std::shared_ptr<FeatureMeasureXYZ> feature_xyz_point)
{
    std::shared_ptr<CostXYZfunction> edge(new CostXYZfunction(pts));
    std::vector<std::shared_ptr<Vertex> > edge_vertex;
    // 关键帧位姿在前、3D点在后
    edge_vertex.push_back(pose);
    edge_vertex.push_back(feature_xyz_point);
    edge->SetVertex(edge_vertex);
    // Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    // Eigen::Vector3d t = Eigen::Vector3d::Zero();
    // edge->SetTranslationImuFromCamera(q, t);

    AddEdge(edge);
}
void BA_problem::addStereoFeatureOneFtwoCResidual(Eigen::Vector3d pts_i, Eigen::Vector3d pts_j, std::shared_ptr<Vertex> feature_point)
{
    std::shared_ptr<CostOneFrameTwoCamFunction> edge(new CostOneFrameTwoCamFunction(pts_i, pts_j));
    std::vector<std::shared_ptr<BaSolver::Vertex>> edge_vertex;
    //添加双目视觉约束，节点的顺序为pose_Fea
    edge_vertex.push_back(feature_point);
    edge->SetVertex(edge_vertex);

    edge->SetInformation(project_sqrt_info_);
    edge->SetLossFunction(lossfunction_);
    AddEdge(edge);
}
void BA_problem::addStereoFeatureTwoFtwoCResidual(Eigen::Vector3d pts_i, Eigen::Vector3d pts_j, std::shared_ptr<Vertex> pose_i, std::shared_ptr<Vertex> pose_j, std::shared_ptr<Vertex> feature_point)
{
    std::shared_ptr<CostTwoFrameTwoCamFunction> edge(new CostTwoFrameTwoCamFunction(pts_i, pts_j));
    std::vector<std::shared_ptr<BaSolver::Vertex>> edge_vertex;
    edge_vertex.push_back(pose_i);
    edge_vertex.push_back(pose_j);
  
    edge_vertex.push_back(feature_point);
    edge->SetVertex(edge_vertex);

    edge->SetInformation(project_sqrt_info_);
    edge->SetLossFunction(lossfunction_);
    AddEdge(edge);
}
void BA_problem::addPriorResidualBlock(std::shared_ptr<Edge> priorEdge)
{
    AddEdge(priorEdge);
}
bool BA_problem::AddVertex(std::shared_ptr<Vertex> vertex) 
{
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
        return false;
    } else {
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
    }
    return true;
}

void BA_problem::AddOrderingSLAM(std::shared_ptr<Vertex> v) 
{
    if (IsPoseVertex(v)) {
        v->SetOrderingId(ordering_poses_);
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
        int dim = v->LocalDimension();
        for(int i = ordering_poses_; i < ordering_poses_ + dim; i ++)
        {
            stRow[i] = ordering_poses_ + dim;
        }
        ordering_poses_ += dim;

    } else if (IsLandmarkVertex(v)) {
        v->SetOrderingId(ordering_landmarks_);
        ordering_landmarks_ += v->LocalDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    }
}
bool BA_problem::IsPoseVertex(std::shared_ptr<Vertex> v) 
{
    string type = v->TypeInfo();
    return type == string("Pose") ||
            type == string("Motion");
}

bool BA_problem::IsLandmarkVertex(std::shared_ptr<Vertex> v) 
{
    string type = v->TypeInfo();
    return type == string("FeatureMeasureXYZ") ||
           type == string("FeatureMeasure");
}

bool BA_problem::AddEdge(shared_ptr<Edge> edge)
{
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
        return false;
    }
    return true;
}
bool BA_problem::solve() 
{

   
    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "\nCannot solve BA_problem without edges or verticies" << std::endl;
        return false;
    }
    TicToc t_solve;
    first_iter_ = true;
    // 统计优化变量的维数，为构建 H 矩阵做准备
    InitProblem();
    SetOrdering();

    // 遍历edge, 构建 H 矩阵
    MakeHessian();
    // LM 初始化
    ComputeLambdaInitLM();

    // LM 算法迭代求解
    bool stop = false;
    int iter = 0;
    double last_chi_ = 1e20;
    while (!stop && (iter < max_iteration_)) {
        iter++;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        while(!oneStepSuccess && false_cnt < 3)
        {
            // 第四步，解线性方程
            SolveLinearSystem();
            
            // 第五步，更新状态量
            UpdateStates();
            delta_norm = std::sqrt(squaredNorm(delta_x, ordering_generic_));
            if(progress_to_stdout)
            {
                std::cout << "iter : " << iter << " , currentChi : " << currentChi_ << " , delta_norm : " << delta_norm << std::endl;
            }
            // 优化退出条件1：当前优化变量的变化量的模长小于是上一次优化变量模长的1e-8
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
            if(delta_norm < 1e-8 * std::sqrt(x_norm))
            {
                stop = true;
                break;
            }
            if(currentLambda_ > 1e32) {
                stop = true;
                break;
            }
            //第六步，计算更新后的残差以及计算rho所需要的分子分母
            ComputeModelCostChange();
            //第七步，判断当前迭代是否合法
            oneStepSuccess = IsGoodStepInLM();
            // 后续处理，
            if (oneStepSuccess) {
                if(iter >= max_iteration_) break;
                MakeHessian();
                false_cnt = 0;
            } else {
                false_cnt ++;
                RollbackStates();   // 误差没下降，回滚
            }
        }
        // 优化退出条件2： currentChi_ 跟第一次的 chi2 相比，下降了 1e6 倍则退出
        if(last_chi_ - currentChi_ < 1e-6 * last_chi_)
        {
            stop = true;
        }
        last_chi_ = currentChi_;
    }
    t_solve_cost = t_solve.toc();
    return true;
}
void BA_problem::SetOrdering() 
{

    TicToc t_setorder;
    // 每次重新计数
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;

    int sz = (cnt_ext_ + cnt_pose_) * 6 + cnt_motion_ * 9;
    stRow = new int[sz];
    
    // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
    for (auto vertex: verticies_) 
    {
        ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量总维数
        AddOrderingSLAM(vertex.second);
    }
    
    // 这里要把 landmark 的 ordering 加上 pose 的数量，就保持了 landmark 在后,而 pose 在前
    ulong all_pose_dimension = ordering_poses_;
    for (auto landmarkVertex : idx_landmark_vertices_) 
    {
        landmarkVertex.second->SetOrderingId(
            landmarkVertex.second->OrderingId() + all_pose_dimension
        );
    }

    ext_size_ = cnt_ext_ * 6;
    pose_size_ = cnt_pose_ * 6;
    motion_size_ = cnt_motion_ * 9;
    landmarks_size_ = ordering_landmarks_;
   
    Hpp = new double[ordering_poses_ * ordering_poses_];
    Hpm = new double[pose_size_ * ordering_landmarks_];
    Hmp = new double[ordering_landmarks_ * pose_size_ ];
    Hmm = new double[ordering_landmarks_ * ordering_landmarks_];
    Hmm_inv = new double[ordering_landmarks_ * ordering_landmarks_];
    tempH = new double[pose_size_ * landmarks_size_];

    b = new double[ordering_generic_];
    diagonal_ = new double[ordering_generic_];
    D_ = new double[ordering_generic_];
    delta_x = new double[ordering_generic_];
    t_setordering_cost += t_setorder.toc();
}

bool BA_problem::CheckOrdering() 
{
    int current_ordering = 0;
    for (auto v: idx_pose_vertices_) {
        assert(v.second->OrderingId() == current_ordering);
        current_ordering += v.second->LocalDimension();
    }

    for (auto v: idx_landmark_vertices_) {
        assert(v.second->OrderingId() == current_ordering);
        current_ordering += v.second->LocalDimension();
    }
    return true;
}
void BA_problem::preResidualJacobian()
{
    TicToc t_p;
    if(pointType_ == PointType::POINT_INVERSE_DEPTH)
    {

        double const *param_ext = verticies_[extToVertex_[0]]->Parameters();
        Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
        Vec3 tic(param_ext[0], param_ext[1], param_ext[2]);
        qic.normalized();
        Qd qic2;
        Vec3 tic2;
        if(STEREO)
        {
            double const *param_ext2 = verticies_[extToVertex_[1]]->Parameters();
            qic2 = Qd(param_ext2[6], param_ext2[3], param_ext2[4], param_ext2[5]);
            qic2.normalized();
            tic2 = Vec3(param_ext2[0], param_ext2[1], param_ext2[2]);
            
            Qd qc2_c1 = qic2.inverse() * qic;
            R_rc_lc = qc2_c1.toRotationMatrix();
            t_rc_lc = qic2.inverse() * (tic - tic2);
            
        }
        for(int i = 0; i < WINDOW_SIZE + 1; i ++)
        {
            double const *param_i = verticies_[poseToVertex_[i]]->Parameters();
            Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
            Vec3 Pi(param_i[0], param_i[1], param_i[2]);
            Qi.normalized();
            for(int j = i + 1; j < WINDOW_SIZE + 1; j ++)
            {
                double const *param_j = verticies_[poseToVertex_[j]]->Parameters();
                Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
                Vec3 Pj(param_j[0], param_j[1], param_j[2]);
                Qj.normalized();

                Qd Qbji = Qj.inverse() * Qi * qic;
                Qd Qji = qic.inverse() * Qbji;
                Qd Qjw = qic.inverse() * Qj.inverse();
                Qd Qjbi = Qjw * Qi;
                Vec3 tbji = Qj.inverse() * ((Qi * tic + Pi) - Pj);
                Vec3 tji = qic.inverse() * (tbji - tic);

                R_cj_ci[i][j] = Qji.toRotationMatrix();
                R_cj_w[i][j] = Qjw.toRotationMatrix();
                R_bj_ci[i][j] = Qbji.toRotationMatrix();
                R_cj_bi[i][j] = Qjbi.toRotationMatrix();
                t_bj_ci[i][j] = tbji;
                t_cj_ci[i][j] = tji;
                if(STEREO)
                {
                    Qd Qrcjlci = qic2.inverse() * Qbji;
                    Vec3 trcjlci = qic2.inverse() * (tbji - tic2);
                    Qd Qrcjw = qic2.inverse() * Qj.inverse();
                    Qd Qrcjbi = Qrcjw * Qi;
                    R_rcj_lci[i][j] = Qrcjlci.toRotationMatrix();
                    t_rcj_lci[i][j] = trcjlci;
                    R_rcj_w[i][j] = Qrcjw.toRotationMatrix();
                    R_rcj_bi[i][j] = Qrcjbi.toRotationMatrix();
                }
            }
        }
    }
    else 
    {

    }
    
    pre_cost += t_p.toc();
}


void BA_problem::MakeHessian()
{
    //预处理视觉约束中的变换
    preResidualJacobian();

    ulong size = ordering_generic_;
    memset(Hpp, 0, sizeof(double)*(ordering_poses_ * ordering_poses_));
    memset(Hpm, 0, sizeof(double)*(pose_size_ * ordering_landmarks_));
    memset(Hmp, 0, sizeof(double)*(ordering_landmarks_ * pose_size_));
    memset(Hmm, 0, sizeof(double)*(ordering_landmarks_ * ordering_landmarks_));
    memset(Hmm_inv, 0, sizeof(double) * (ordering_landmarks_ * ordering_landmarks_));
    memset(b, 0, sizeof(double) * (size));
    TicToc t_h;
    for(auto &edge : edges_)
    {
        edge.second->ComputeJacobians();
    }
    memset(delta_x, 0, sizeof(double) * ordering_generic_);
    t_hessian_cost += t_h.toc();   
}


void BA_problem::ComputeModelCostChange()
{ 
    preResidualJacobian();
    TicToc t_cost;
    tempChi_ = 0;
    for(auto &edge : edges_)
    {
        edge.second->ComputeResidual();
        edge.second->ComputeChi2();
        tempChi_ += edge.second->RobustChi2();
    }
    tempChi_ *= 0.5;
   
    Eigen::Map<Eigen::VectorXd> dd(delta_x, ordering_generic_);
    Eigen::Map<Eigen::VectorXd> bb(b, ordering_generic_);
    model_cost_change_ = 0.5 * dd.dot(currentLambda_ * dd + bb);
    t_res_cost += t_cost.toc();
   
}

/*
 * Solve Hx = b
 */
void BA_problem::SolveLinearSystem() {

    TicToc t_solve;

    if(problemType_ == ProblemType::SLAM_BAPROBLEM) 
    {
        //BA 问题求解
        if(pointType_ == PointType::POINT_INVERSE_DEPTH) 
        {
            // 路标点为逆深度点
            int size  = ordering_poses_;
            //填充左下角
            for(int i = 0; i < size; i ++)
            {
                for(int j = stRow[i]; j < size; j ++)
                {
                    Hpp[j * size + i] = Hpp[i * size + j];
                }
            }
            transpose(Hpm, pose_size_, ordering_landmarks_, Hmp);
        // 计算D
            double maxDiagonal = 0;
            min_diagonal_ = 1e-6;
            max_diagonal_ = 1e32;
            //左上角
            for(int i = 0; i < ordering_poses_; i ++)
            {
                diagonal_[i] = 0;
                double x = Hpp[i * size + i];
                if(x == 0.0) continue;
                diagonal_[i] = std::sqrt(std::min(std::max(x, min_diagonal_), max_diagonal_));
                diagonal_[i] = std::sqrt(currentLambda_) * diagonal_[i];
            }
            //右下角
            for(int i = ordering_poses_; i < ordering_generic_; i ++)
            {
                int ii = i - ordering_poses_;
                double x = Hmm[ii * landmarks_size_ + ii];
                diagonal_[i] = std::sqrt(std::min(std::max(x, min_diagonal_), max_diagonal_));
                diagonal_[i] = std::sqrt(currentLambda_) * diagonal_[i];
            }
            for(int i = 0; i < ordering_generic_; i ++)
            {
                D_[i] = 0;
                if(diagonal_[i] == 0) continue;
                D_[i] = diagonal_[i] * diagonal_[i];
            }
        //将D加入信息矩阵中
            //左上角
            for(int i = 0; i < ordering_poses_; i ++)
            {
                Hpp[i * size + i] += D_[i];
            }
            //右下角
            for(int i = ordering_poses_; i < ordering_generic_; i ++)
            {
                int ii = i - ordering_poses_;
                Hmm[ii * ordering_landmarks_ + ii] += D_[i];
            }
            //计算Hmm的逆、Hmm_inv
            for(int i = 0; i < ordering_landmarks_; i ++)
            {
                int x = i * ordering_landmarks_ + i;
                Hmm_inv[x] = 1.0 / Hmm[x];
            }
            //计算tempH
            for(int i = 0; i < pose_size_; i ++)
            {
                for(int j = 0; j <ordering_landmarks_; j ++)
                {
                    int x = i * ordering_landmarks_ + j;
                    tempH[x] = Hpm[x] * Hmm_inv[j * ordering_landmarks_ + j];
                }
            }
        //构建舒尔补矩阵
            // for(int i = 0; i < pose_size_; i ++)
            // {
            //     int ii = i + ext_size_;
            //     for(int j = 0; j < pose_size_; j ++)
            //     {
            //         int jj = j + ext_size_;
            //         for(int k = 0; k < ordering_landmarks_; k ++)
            //         {
            //             Hpp[ii * ordering_poses_ + jj] -= Hpm[i * landmarks_size_ + k] * Hmp[k * pose_size_ + j] / Hmm[k * landmarks_size_ + k];
            //         }
            //     }
            // }
            double *bb = new double[ordering_generic_];
            memcpy(bb, b, sizeof(double) * ordering_generic_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHpp(Hpp, ordering_poses_, ordering_poses_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHpm(Hpm, pose_size_, ordering_landmarks_); 
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHmp(Hmp, ordering_landmarks_, pose_size_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EtempH(tempH, pose_size_, ordering_landmarks_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHmm_inv(Hmm_inv, ordering_landmarks_, ordering_landmarks_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHmm(Hmm, ordering_landmarks_, ordering_landmarks_);
            
            Eigen::Map<VectorXd> bpp(bb, ordering_poses_);
            Eigen::Map<VectorXd> bmm(bb + ordering_poses_, ordering_landmarks_);
            memset(delta_x, 0, sizeof(double) * ordering_generic_);
            Eigen::Map<VectorXd> delta_x_pp(delta_x, ordering_poses_);
            Eigen::Map<VectorXd> delta_x_mm(delta_x + ordering_poses_, ordering_landmarks_);


            EHpp.block(ext_size_, ext_size_, pose_size_, pose_size_).noalias() -= EtempH * EHmp;
            bpp.segment(ext_size_, pose_size_).noalias() -= EtempH * bmm;

            delta_x_pp = EHpp.ldlt().solve(bpp);
            auto bbmm = EHmp * delta_x_pp.segment(ext_size_, pose_size_);
            delta_x_mm = EHmm_inv * (bmm - bbmm);
            delete[] bb;
        }
        else
        {
            // 路标点为3D点
    
            int size  = ordering_poses_;
        
            transpose(Hpm, pose_size_, ordering_landmarks_, Hmp);
        // 计算D
            double maxDiagonal = 0;
            min_diagonal_ = 1e-6;
            max_diagonal_ = 1e32;
            //左上角
            for(int i = 0; i < ordering_poses_; i ++)
            {
                diagonal_[i] = 0;
                double x = Hpp[i * size + i];
                if(x == 0.0) continue;
                diagonal_[i] = std::sqrt(std::min(std::max(x, min_diagonal_), max_diagonal_));
                diagonal_[i] = std::sqrt(currentLambda_) * diagonal_[i];
            }
            //右下角
            for(int i = ordering_poses_; i < ordering_generic_; i ++)
            {
                int ii = i - ordering_poses_;
                double x = Hmm[ii * landmarks_size_ + ii];
                diagonal_[i] = std::sqrt(std::min(std::max(x, min_diagonal_), max_diagonal_));
                diagonal_[i] = std::sqrt(currentLambda_) * diagonal_[i];
            }
            for(int i = 0; i < ordering_generic_; i ++)
            {
                D_[i] = 0;
                if(diagonal_[i] == 0) continue;
                D_[i] = diagonal_[i] * diagonal_[i];
            }
        //将D加入信息矩阵中
            //左上角
            for(int i = 0; i < ordering_poses_; i ++)
            {
                Hpp[i * size + i] += D_[i];
            }
            //右下角
            for(int i = ordering_poses_; i < ordering_generic_; i ++)
            {
                int ii = i - ordering_poses_;
                Hmm[ii * ordering_landmarks_ + ii] += D_[i];
            }


            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHmm_inv(Hmm_inv, ordering_landmarks_, ordering_landmarks_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHmm(Hmm, ordering_landmarks_, ordering_landmarks_);
            //计算Hmm的逆、Hmm_inv
        
            for (auto landmarkVertex : idx_landmark_vertices_) {
                int idx = landmarkVertex.second->OrderingId() - ordering_poses_;
                int size = landmarkVertex.second->LocalDimension();
                EHmm_inv.block(idx, idx, size, size) = EHmm.block(idx, idx, size, size).inverse();
            }
            //计算tempH
            for(int i = 0; i < pose_size_; i += 6)
            {
                for(int j = 0; j <ordering_landmarks_; j += 3)
                {
                    double tmpA[18], tmpB[9];
                    double tmpC[18];
                    memset(tmpC, 0, sizeof(tmpC));
                    block(Hpm, i, j, 6, 3, ordering_landmarks_, tmpA);
                    block(Hmm_inv, j, j, 3, 3, ordering_landmarks_, tmpB);
                    multiMatrix(tmpA, tmpB, 6, 3, 3, tmpC);
                    setBlock(tempH, i, j, 6, 3, ordering_landmarks_, tmpC);
                }
            }
        //构建舒尔补矩阵
    
            double *bb = new double[ordering_generic_];
            memcpy(bb, b, sizeof(double) * ordering_generic_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHpp(Hpp, ordering_poses_, ordering_poses_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHpm(Hpm, pose_size_, ordering_landmarks_); 
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EHmp(Hmp, ordering_landmarks_, pose_size_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EtempH(tempH, pose_size_, ordering_landmarks_);
            
            
            Eigen::Map<VectorXd> bpp(bb, ordering_poses_);
            Eigen::Map<VectorXd> bmm(bb + ordering_poses_, ordering_landmarks_);
            memset(delta_x, 0, sizeof(double) * ordering_generic_);
            Eigen::Map<VectorXd> delta_x_pp(delta_x, ordering_poses_);
            Eigen::Map<VectorXd> delta_x_mm(delta_x + ordering_poses_, ordering_landmarks_);


            EHpp.block(ext_size_, ext_size_, pose_size_, pose_size_).noalias() -= EtempH * EHmp;
            bpp.segment(ext_size_, pose_size_).noalias() -= EtempH * bmm;

            delta_x_pp = EHpp.ldlt().solve(bpp);
            auto bbmm = EHmp * delta_x_pp.segment(ext_size_, pose_size_);
            delta_x_mm = EHmm_inv * (bmm - bbmm);
            delete[] bb;
        }
        
    }
    else 
    {
        // Pose Graph问题求解
        
    }

    t_linear_solve_cost += t_solve.toc();
}

void BA_problem::UpdateStates() {
    
    TicToc t_update;
    x_norm = 0;
    for (auto vertex: verticies_) {
        if(vertex.second->IsFixed()) continue;
        vertex.second->BackUpParameters();    // 保存上次的估计值
        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        double* delta = delta_x + idx;
        double *para = vertex.second->Parameters();
        int sz = vertex.second->Dimension();
        x_norm += squaredNorm(para, sz);
        vertex.second->Plus(delta);
    }
    t_update_cost += t_update.toc();
}

void BA_problem::RollbackStates() {

    // update vertex
    for (auto vertex: verticies_) {
        if(vertex.second->IsFixed()) continue;
        vertex.second->RollBackParameters();
    }
}

/// LM 算法初始化
void BA_problem::ComputeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = 1e-4;
    currentChi_ = 0.0;
    for (auto edge: edges_) {
        double x = edge.second->RobustChi2();
        currentChi_ += x;
     
    }
    currentChi_ *= 0.5;
}

bool BA_problem::IsGoodStepInLM() {
    TicToc t_judge;
    double scale = model_cost_change_;
    scale += 1e-6;    // 确保scale非零
    double rho = (currentChi_ - tempChi_) / scale;

    if (rho > 0.001 && isfinite(tempChi_))   // 当前迭代满足条件, 误差在下降
    {
        double alpha = 1. - pow((2 * rho - 1), 3);
        double scaleFactor = (std::max)(1. / 3., alpha);
        currentLambda_ *= scaleFactor;
        ni_ = 2;
        currentChi_ = tempChi_;
        t_judge_cost += t_judge.toc();
        return true;
    } else {
        currentLambda_ *= ni_;
        ni_ *= 2;
        t_judge_cost += t_judge.toc();
        return false;
    }
}

}
