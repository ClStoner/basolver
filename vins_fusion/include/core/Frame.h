/**
 * @file Frame.h
 * @author OkayManXi (786783021@qq.com)
 * @brief Tracker类中视觉前端追踪的Frame视觉帧数据结构
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

namespace core
{
/**
 * @brief 视觉帧Frame数据结构，用于视觉前端帧间追踪点数据结构
 *
 */
class Frame
{
  public:
    POINTER_TYPEDEFS(Frame);

    // default
    Frame() : frameID(0), timeStamp(0) {}

    // Mono
    /**
     * @brief Construct a new Frame object Mono
     *
     * @param timeStamp 视觉时间戳
     * @param img1 左目
     */
    Frame(const double timeStamp, const cv::Mat &img1)
        : Stereo(params.Stereo), Depth(params.Depth), OpticalFlow(params.OpticalFlow), frameID(frameIDCnt++), timeStamp(timeStamp),
          leftImg(img1.clone()), originImg(img1.clone())
    {
        // equalize
        if (params.clahe)
        {
            clahe->apply(leftImg, leftImg);
        }
    }

    // Stereo
    /**
     * @brief Construct a new Frame object Stereo / Mono + Depth / Mono + OF
     *
     * @param timeStamp 视觉时间戳
     * @param img1 左目
     * @param img2 右目
     */
    Frame(const double timeStamp, const cv::Mat &img1, const cv::Mat &img2)
        : Stereo(params.Stereo), Depth(params.Depth), OpticalFlow(params.OpticalFlow), frameID(frameIDCnt++), timeStamp(timeStamp),
          leftImg(img1.clone()), originImg(img1.clone())
    {
        if (Stereo)
            rightImg = img2;
        else if (Depth)
            depthImg = img2;
        else if (OpticalFlow)
            p2cImg = img2;
        else
            LOG(ERROR) << "Frame State Error";

        // equalize
        if (params.clahe)
        {
            clahe->apply(leftImg, leftImg);
            if (Stereo)
                clahe->apply(rightImg, rightImg);
        }
    }

    /**
     * @brief Construct a new Frame object Stereo + Depth / Mono + Depth + OF
     *
     * @param timeStamp 时间戳
     * @param img1 左目
     * @param img2 右目
     * @param img3 深度
     */
    Frame(const double timeStamp, const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &img3)
        : Stereo(params.Stereo), Depth(params.Depth), OpticalFlow(params.OpticalFlow), frameID(frameIDCnt++), timeStamp(timeStamp),
          leftImg(img1.clone()), originImg(img1.clone())
    {
        if (Stereo)
        {
            rightImg = img2;
            depthImg = img3;
        }
        else if (Depth && OpticalFlow)
        {
            depthImg = img2;
            p2cImg = img3;
        }
        else
            LOG(ERROR) << "Frame State Error";

        // equalize
        if (params.clahe)
        {
            clahe->apply(leftImg, leftImg);
            if (Stereo)
                clahe->apply(rightImg, rightImg);
        }
    }

    /**
     * @brief Construct a new Frame object Stereo + OpticalFlow
     *
     * @param timeStamp 时间戳
     * @param img1 左目
     * @param img2 右目
     * @param img3 前后光流
     * @param img4 左右光流
     */
    Frame(const double timeStamp, const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &img3, const cv::Mat &img4)
        : Stereo(params.Stereo), Depth(params.Depth), OpticalFlow(params.OpticalFlow), frameID(frameIDCnt++), timeStamp(timeStamp),
          leftImg(img1.clone()), rightImg(img2.clone()), p2cImg(img3.clone()), l2rImg(img4.clone()), originImg(img1.clone())
    {
        // equalize
        if (params.clahe)
        {
            clahe->apply(leftImg, leftImg);
            clahe->apply(rightImg, rightImg);
        }
    }

    /**
     * @brief Construct a new Frame object Stereo + OpticalFlow + Depth
     *
     * @param timeStamp 时间戳
     * @param img1 左目
     * @param img2 右目
     * @param img3 深度
     * @param img4 前后光流
     * @param img5 左右光流
     */
    Frame(const double timeStamp, const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &img3, const cv::Mat &img4, const cv::Mat &img5)
        : Stereo(params.Stereo), Depth(params.Depth), OpticalFlow(params.OpticalFlow), frameID(frameIDCnt++), timeStamp(timeStamp),
          leftImg(img1.clone()), rightImg(img2.clone()), depthImg(img3.clone()), p2cImg(img4.clone()), l2rImg(img5.clone()), originImg(img1.clone())
    {
        // equalize
        if (params.clahe)
        {
            clahe->apply(leftImg, leftImg);
            clahe->apply(rightImg, rightImg);
        }
    }

    // copy
    Frame(const Frame &frame) : frameID(frame.frameID), timeStamp(frame.timeStamp)
    {
        // flag
        this->Stereo = frame.Stereo;
        this->Depth = frame.Depth;
        this->OpticalFlow = frame.OpticalFlow;

        // img
        this->leftImg = frame.leftImg.clone();
        this->rightImg = frame.rightImg.clone();
        this->depthImg = frame.depthImg.clone();
        this->p2cImg = frame.p2cImg.clone();
        this->l2rImg = frame.l2rImg.clone();
        this->originImg = frame.originImg.clone();

        // feature
        this->points = frame.points;
    }

    // move
    Frame(Frame &&frame) : frameID(frame.frameID), timeStamp(frame.timeStamp)
    {
        // img
        this->Stereo = std::move(frame.Stereo);
        this->Depth = std::move(frame.Depth);
        this->OpticalFlow = std::move(frame.OpticalFlow);

        // img
        this->leftImg = std::move(frame.leftImg);
        this->rightImg = std::move(frame.rightImg);
        this->depthImg = std::move(frame.depthImg);
        this->p2cImg = std::move(frame.p2cImg);
        this->l2rImg = std::move(frame.l2rImg);
        this->originImg = std::move(frame.originImg);

        // feature
        this->points = std::move(frame.points);
    }

    Frame &operator=(const Frame &rhs)
    {
        if (this == &rhs)
            return *this;

        // flag
        this->Stereo = rhs.Stereo;
        this->Depth = rhs.Depth;
        this->OpticalFlow = rhs.OpticalFlow;

        // img
        this->leftImg = rhs.leftImg.clone();
        this->rightImg = rhs.rightImg.clone();
        this->depthImg = rhs.depthImg.clone();
        this->p2cImg = rhs.p2cImg.clone();
        this->l2rImg = rhs.l2rImg.clone();
        this->originImg = rhs.originImg.clone();

        // feature
        this->points = rhs.points;

        return *this;
    }

    /**
     * @brief 计算特征点归一化坐标
     *
     */
    void undistortedPoints();

    /**
     * @brief 计算帧间特帧点运动速度
     *
     * @param prev 参考帧
     */
    void updatePointsVelocity(const Frame::Ptr &prev = nullptr);

    /**
     * @brief 依据右目或深度图，进行左右目匹配外点去除
     *
     */
    void addPointRightObservation();

    /**
     * @brief 检查特征点ID是否有重复
     *
     */
    void checkPoints();

    // static params
    /// frame counter
    static long frameIDCnt;
    /// point counter
    static long pointIDCnt;
    /// camera model for left and right camera
    static camodocal::CameraPtr leftCam, rightCam;
    /// clahe
    static cv::Ptr<cv::CLAHE> clahe;

    // flag
    /// Stereo flog
    bool Stereo;
    /// Depth flag
    bool Depth;
    /// OpticalFlow flag
    bool OpticalFlow;

    // base params
    /// Frame id
    const long frameID;
    /// Frame timestamp
    const double timeStamp;
    /// Frame intput left img
    cv::Mat leftImg;
    /// Frame input right img
    cv::Mat rightImg;
    /// Frame input depth img
    cv::Mat depthImg;
    /// Frame input prev2curr and left2right opticalflow img
    cv::Mat p2cImg, l2rImg;
    /// Frame orgin left img
    cv::Mat originImg;

    // keypoints
    /// Frame track points
    std::vector<PointInfo> points;
};
}  // namespace core
