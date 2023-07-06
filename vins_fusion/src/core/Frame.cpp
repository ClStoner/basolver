#include "core/Frame.h"

namespace core
{

long Frame::frameIDCnt;
long Frame::pointIDCnt;
camodocal::CameraPtr Frame::leftCam;
camodocal::CameraPtr Frame::rightCam;
cv::Ptr<cv::CLAHE> Frame::clahe;

void Frame::undistortedPoints()
{
    for (auto &pt : points)
    {
        Eigen::Vector3d p3D;
        leftCam->liftProjective(pt.p2D, p3D);
        pt.p3D = p3D / p3D.z();

        if (pt.stereo)
        {
            rightCam->liftProjective(pt.p2DRight, p3D);
            pt.p3DRight = p3D / p3D.z();
        }
    }
}

void Frame::updatePointsVelocity(const Frame::Ptr &prev)
{
    if (!prev || prev->points.empty())
    {
        for (auto &pt : points)
        {
            pt.velocity = Eigen::Vector2d::Zero();
            if (pt.stereo)
                pt.velocityRight = Eigen::Vector2d::Zero();
        }

        return;
    }

    double dt = this->timeStamp - prev->timeStamp;
    std::map<long, PointInfo> prevIDPoints;
    for (auto &pt : prev->points)
        prevIDPoints.emplace(pt.id, pt);

    for (auto &pt : points)
    {
        auto iter = prevIDPoints.find(pt.id);
        if (iter != prevIDPoints.end())
        {
            Eigen::Vector3d velocity = (pt.p3D - iter->second.p3D) / dt;
            pt.velocity = velocity.head(2);

            // LOG(INFO) << "v: " << pt.velocity.transpose() << " prev: " << iter->second.p2D.transpose() << " curr: " << pt.p2D.transpose();

            if (pt.stereo && iter->second.stereo)
            {
                Eigen::Vector3d velocityRight = (pt.p3DRight - iter->second.p3DRight) / dt;
                pt.velocityRight = velocityRight.head(2);
                // LOG(INFO) << "vRight: " << pt.velocityRight.transpose() << " prev: " << iter->second.p2DRight.transpose()
                //           << " curr: " << pt.p2DRight.transpose();
            }
        }
        else
        {
            pt.velocity = Eigen::Vector2d::Zero();
            if (pt.stereo)
                pt.velocityRight = Eigen::Vector2d::Zero();
        }
    }
}

void Frame::addPointRightObservation()
{
    if (!!Depth || !Stereo)
        return;

    CHECK_EQ(depthImg.type(), CV_32F);

    // cv::Mat showImg = leftImg.clone();
    // cv::resize(showImg, showImg, cv::Size(2 * showImg.cols, 2 * showImg.rows));
    // if (showImg.channels() != 3)
    //     cv::cvtColor(showImg, showImg, cv::COLOR_GRAY2RGB);
    // cv::Mat showImg_ = showImg.clone();
    const double bf = params.baseLine;  // pointUn focallength = 1
    for (auto &pt : points)
    {
        const double d = depthImg.at<float>(pt.p2D.y(), pt.p2D.x());
        if (d > params.depthTh)
            continue;

        if (pt.stereo)
        {
            Eigen::Vector3d p3DDepth{pt.p3D.x() - bf / d, pt.p3D.y(), 1};
            Eigen::Vector3d delta = p3DDepth - pt.p3DRight;
            double norm = delta.norm();

            if (params.focalLength * norm > 1.0)
                pt.stereo = false;
            else
                pt.depth = d;
        }
    }
}

void Frame::checkPoints()
{
    // check point num
    std::set<long> idSet;
    for (auto &pt : points)
    {
        CHECK_EQ(idSet.count(pt.id), 0);
        idSet.insert(pt.id);
    }
}

}  // namespace core