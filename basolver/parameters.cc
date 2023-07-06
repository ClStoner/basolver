#include "include/parameters.h"


namespace BaSolver
{
double ACC_N, ACC_W;
double GYR_N, GYR_W;
Eigen::Vector3d G;
int NUM_OF_CAM;
int NUM_OF_THREADS;
bool STEREO;

double *b;
double *Hpp;
double *Hpm;
double *Hmp;
double *Hmm;
double *Hmm_inv;
double *tempH;
double *delta_x;
int WINDOW_SIZE;
std::vector<std::vector<Eigen::Matrix3d>> R_cj_ci;
std::vector<std::vector<Eigen::Vector3d>> t_cj_ci;
std::vector<std::vector<Eigen::Matrix3d>> R_cj_w;
std::vector<std::vector<Eigen::Matrix3d>> R_bj_ci;
std::vector<std::vector<Eigen::Vector3d>> t_bj_ci;
std::vector<std::vector<Eigen::Matrix3d>> R_cj_bi;

std::vector<std::vector<Eigen::Matrix3d>> R_rcj_lci;
std::vector<std::vector<Eigen::Vector3d>> t_rcj_lci;
std::vector<std::vector<Eigen::Matrix3d>> R_rcj_w;
std::vector<std::vector<Eigen::Matrix3d>> R_rcj_bi;

std::vector<Eigen::Matrix3d> Ric;
std::vector<Eigen::Matrix3d> Rci;
std::vector<Eigen::Vector3d> tic;
std::vector<Eigen::Vector3d> tci;

Eigen::Matrix3d R_rc_lc;
Eigen::Vector3d t_rc_lc;

std::unordered_map<int, int> pointToVertex_;
std::unordered_map<int, int> vertexToPoint_;

std::unordered_map<int, int> poseToVertex_;
std::unordered_map<int, int> vertexToPose_;

std::unordered_map<int, int> extToVertex_;
std::unordered_map<int, int> vertexToExt_;


std::unordered_map<int, std::pair<int, int>> startFrame_;




int ext_size_;
int pose_size_;
int motion_size_;
int landmarks_size_;
int ordering_poses_;
int ordering_landmarks_;
int ordering_generic_ ;
void readParameters(std::string config_file)
{
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }
    ACC_N = fsSettings["accN"];
    ACC_W = fsSettings["accW"];
    GYR_N = fsSettings["gyrN"];
    GYR_W = fsSettings["gyrW"];

    G.z() = fsSettings["gNorm"];
    NUM_OF_CAM = fsSettings["num_of_cam"];
    NUM_OF_THREADS = fsSettings["num_of_thread"];
    WINDOW_SIZE = fsSettings["window_size"];
    std::cout << "ACC_N : " << ACC_N << ", ACC_W : " << ACC_W << ", GYR_N : " << GYR_N << " , GYR_W : " << GYR_W << std::endl;
}

}