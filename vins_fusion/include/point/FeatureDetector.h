/**
 * @file FeatureDetector.h
 * @author OkayManXi (786783021@qq.com)
 * @brief Harris、Fasst特征提取算法实现类
 * @version 0.1
 * @date 2022-08-11
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "utils/Marco.h"
#include "utils/Utility.h"
#include "utils/Params.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

/**
 * @brief 四叉树节点，内部存储节点内角点
 * 参考ORB-SLAM3 ORBextractor.cc DistributeOctTree()
 */
class OctTreeNode
{
  public:
    POINTER_TYPEDEFS(OctTreeNode);
    OctTreeNode();
    /**
     * @brief 四叉树内部分裂
     *
     * @param n1 字节点1
     * @param n2 字节点2
     * @param n3 字节点3
     * @param n4 字节点4
     */
    void divideNode(OctTreeNode &n1, OctTreeNode &n2, OctTreeNode &n3, OctTreeNode &n4);
    /**
     * @brief 可视化节点
     * @refer ORB-SLAM ORBextractor
     * @param _img
     */
    void drawNode(cv::Mat &_img);
    /// OctTreeNode can reproduce
    bool reproduce;
    /// OctTreeNode inner kps
    std::vector<cv::KeyPoint> kps;
    /// OctTreeNode 4 edge point location
    cv::Point2i UL, UR, BL, BR;
    /// OctTreeNode iter
    std::list<Ptr>::iterator lit;
};
/**
 * @brief Feature提取接口类
 *
 */
class FeatureDetector
{
  protected:
    /// FeatureDetector detect max feature num
    int max_feature;
    /// FeatureDetector thereshold
    const int thereshold;
    /// FeatureDetector img height
    const int img_height;
    /// FeatureDetector img width
    const int img_width;

    /// FeatureDetector curr img, show img and mask img
    cv::Mat curr_img, show_img, mask;
    /// FeatureDetector detected kps
    std::vector<cv::KeyPoint> keypoints;

  public:
    POINTER_TYPEDEFS(FeatureDetector);
    FeatureDetector(int _max_feature, cv::Size _img_size, int _thereshold);

    /**
     * @brief 可视化角点
     *
     */
    virtual void drawKps() = 0;
    /**
     * @brief 清空内部状态
     *
     */
    virtual void clear() = 0;
    /**
     * @brief 设置最大提点数量
     *
     * @param _max_feature
     */
    virtual void setMaxFeature(int _max_feature) = 0;
    /**
     * @brief 提取角点
     *
     * @param _img 输入图像
     * @param _kps 输出角点
     * @param _mask 输入mask
     */
    virtual void detect(cv::Mat &_img, std::vector<cv::KeyPoint> &_kps, cv::Mat &_mask) = 0;
};

/**
 * @brief Harris提取
 *
 */
class HarrisDetector : public FeatureDetector
{
  protected:
    // harirs params
    /// HarrisDetector feature detect min dist
    const int min_dist;

    // worker
    /// HarrisDetector inner opencv harris worker
    cv::Ptr<cv::GFTTDetector> harris;

  public:
    POINTER_TYPEDEFS(HarrisDetector);
    HarrisDetector(int _max_feature, cv::Size _img_size, int _thereshold, int _min_dist);

    /**
     * @brief 可视化角点
     *
     */
    virtual void drawKps();
    /**
     * @brief 清空内部状态
     *
     */
    virtual void clear();
    /**
     * @brief 设置最大提点数量
     *
     * @param _max_feature
     */
    virtual void setMaxFeature(int _max_feature);
    /**
     * @brief
     *
     * @param _img 输入图像
     * @param _kps 输出角点
     * @param _mask 输入mask
     */
    virtual void detect(cv::Mat &_img, std::vector<cv::KeyPoint> &_kps, cv::Mat &_mask);
};

/**
 * @brief Fast角点提取
 *  复现点ICRA2021 ”Stereo Visual Inertial Odometry for Robots with Limited Computational Resources“中关于提点部分
 */
class FastDetector : public FeatureDetector
{
  protected:
    // grid fast params
    /// FastDetector feature detect cell size
    const int cell_size;
    /// FastDetector cell cols num
    const int num_cols;
    /// FastDetector cell rows num
    const int num_rows;
    /// FastDetector cell start offset x
    const int offset_X;
    /// FastDetector cell start offset y
    const int offset_Y;
    /// FastDetector feature detect num per cell
    int grid_f_num;

    // worker
    /// FastDetector inner opencv fast worker
    cv::Ptr<cv::FastFeatureDetector> fast;

    // data
    /// FastDetector octtreennodes
    std::list<OctTreeNode::Ptr> OcTtreeNodes;
    /// FastDetector init octtreenods
    std::vector<OctTreeNode::Ptr> InitOctTreeNodes;
    /// FastDetector grid mask flag detect feature
    std::vector<bool> gridMask;
    /// FastDetector grid mask cnt feature has detect
    std::vector<int> gridMaskCnt;

  public:
    POINTER_TYPEDEFS(FastDetector);
    FastDetector(int _max_feature, cv::Size _img_size, int _thereshold, int _cell_size);

    /**
     * @brief 依据角点分布判断网格是否需要提点
     *
     * @param _kps 参考帧角点
     */
    void updateMask(std::vector<cv::KeyPoint> &_kps);
    /**
     * @brief
     *
     * @param _grid_index 网格提点
     */
    void detectGrid(cv::Point2i _grid_index);
    /**
     * @brief 四叉树NMS
     * @refer ORB-SLAM ORBextractor
     */
    void distriOctTree();

    /**
     * @brief 可视化角点
     *
     */
    virtual void drawKps();
    /**
     * @brief 清空内部状态
     *
     */
    virtual void clear();
    /**
     * @brief 设置最大提点数量
     *
     * @param _max_feature
     */
    virtual void setMaxFeature(int _max_feature);
    /**
     * @brief
     *
     * @param _img 输入图像
     * @param _kps 输出角点
     * @param _mask 输入mask
     */
    virtual void detect(cv::Mat &_img, std::vector<cv::KeyPoint> &_kps, cv::Mat &_mask);
    /**
     * @brief
     * @refer ICRA2021 ”Stereo Visual Inertial Odometry for Robots with Limited Computational Resources“
     * @param _img 输入图像
     * @param _kps 输出角点
     * @param _mask 输入mask
     * @param _prev_kps 参考帧角点
     */
    void detect(cv::Mat &_img, std::vector<cv::KeyPoint> &_kps, cv::Mat &_mask, std::vector<cv::KeyPoint> &_prev_kps);
};