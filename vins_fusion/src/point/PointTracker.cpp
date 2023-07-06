#include "point/PointTracker.h"

using namespace core;

PointTracker::PointTracker()
    : cnt(0), logFreq(params.logFreq), Stereo(params.Stereo), imgHeight(params.imgHeight), imgWidth(params.imgWidth), focalLength(params.focalLength),
      borderSize(params.borderSize), pointExtractType(params.pointExtractType), pointExtractMaxNum(params.pointExtractMaxNum),
      pointExtractMinDist(params.pointExtractMinDist), pointMatchRansacTh(params.pointMatchRansacTh), Ric(params.TicLeft.block(0, 0, 3, 3)),
      hasPredict(false), camera(params.leftCam)
{
    LOG(INFO) << "PointTracker create";
    if (pointExtractType == PointExtractFlag::FAST)
        fast = std::make_shared<FastDetector>(0, cv::Size{params.imgWidth, params.imgHeight}, 20, 40);
    else if (pointExtractType == PointExtractFlag::Harris)
        harris = std::make_shared<HarrisDetector>(0, cv::Size{params.imgWidth, params.imgHeight}, 20, 30);

    // camera = params.leftCam;
}

// TODO inference
void PointTracker::inference(const Frame::Ptr &prev, const Frame::Ptr &curr, const bool isKeyframe)
{
    utility::TicToc td;

    // klt tracker
    if (!params.OpticalFlow)
        this->matchByLKOpticFlow(prev, curr);
    else
        this->matchByFBOpticFlow(prev, curr);

    if (isKeyframe || curr->points.size() < 0.6 * pointExtractMaxNum)
    {
        // LOG(INFO) << "isKeyframe: " << isKeyframe << " point size: " << curr->points.size();
        this->rejectWithF(prev, curr);
        this->extractPoints(prev, curr);
    }

    if (Stereo && curr->Stereo)
    {
        if (!params.OpticalFlow)
            this->matchRightPointsByLKOpticFlow(curr);
        else
            this->matchRightPointsByFBOpticFlow(curr);
    }

    // update undistort and velocity
    curr->undistortedPoints();
    curr->updatePointsVelocity(prev);
    if (curr->Depth && curr->Stereo)
        curr->addPointRightObservation();
    curr->checkPoints();

    cnt++;
#ifdef TEST_PERF
{
    static double sum_dt = 0;
    static int cnt = 0;
    sum_dt += td.toc();
    cnt++;
    printf("point inference cost: %f\n", sum_dt / cnt);
}
#endif
    LOG_EVERY_N(INFO, logFreq) << "point inference cost: " << td.toc() << "ms";
}

void PointTracker::setPredictPoints(const Frame::Ptr &frame, const Eigen::Matrix3d &R)
{
    if (!frame || frame->points.empty() || !utility::rotIsValid(R))
    {
        hasPredict = false;
        predictPoints.clear();
        return;
    }

    hasPredict = true;
    predictPoints.resize(frame->points.size());
    for (unsigned i = 0; i < frame->points.size(); i++)
    {
        Eigen::Vector3d predictPtUn = frame->points[i].p3D;
        predictPtUn = R * predictPtUn;
        Eigen::Vector2d predictPtUV;
        camera->spaceToPlane(predictPtUn, predictPtUV);
        cv::Point2f pointUV{predictPtUV.x(), predictPtUV.y()};
        if (inBorder(pointUV))
        {
            predictPoints[i] = pointUV;
        }
        else
            predictPoints[i] = cv::Point2f{frame->points[i].p2D.x(), frame->points[i].p2D.y()};
    }
}

cv::Mat PointTracker::drawPoints(const Frame::Ptr &frame, const bool isShow)
{
    auto &points = frame->points;
    if (points.empty())
        return cv::Mat();

    cv::Mat showImg = frame->leftImg.clone();
    if (showImg.channels() == 1)
        cv::cvtColor(showImg, showImg, cv::COLOR_GRAY2RGB);

    // draw predict points
    if (!predictPoints.empty())
    {
        for (unsigned i = 0; i < predictPoints.size(); i++)
        {
            // cv::circle(showImg, points[i], 2, cv::Scalar(0, 255, 0), -1);                     // prev points draw green color
            cv::circle(showImg, predictPoints[i], 2, cv::Scalar(0, 255, 255), -1);  // predict points draw yellow color
            // cv::arrowedLine(showImg, points[i], predictPoints[i], cv::Scalar(255, 0, 0), 1);  // draw arrow from prev points to predict points
        }
        predictPoints.clear();
    }

    // curr points
    for (auto &pt : frame->points)
    {
        double len = std::min(1.0, 1.0 * pt.cnt / 20);
        cv::circle(showImg, cv::Point2d{pt.p2D.x(), pt.p2D.y()}, 2, cv::Scalar(255 * (1 - len), 0, 255 * len), -1);
    }

    if (isShow)
    {
        // cv::imwrite("/root/myGit/mycode/vins_rgbd/src/vins_rgbd/test_rgbd/results/img/lineEtract" + std::to_string(cnt) + ".png", showImg);
        cv::imshow("pointExtract", showImg);
        cv::moveWindow("pointExtract", showImg.cols + 80, 0);
        cv::waitKey(1);
    }
    return showImg;
}

cv::Mat PointTracker::sortPointsByCnt(const Frame::Ptr &frame)
{
    if (!frame || frame->points.empty())
        return cv::Mat();

    auto &points = frame->points;

    const cv::Scalar blackColor{0}, whiteColor{255};
    cv::Mat mask{imgHeight, imgWidth, CV_8UC1, blackColor};
    cv::rectangle(mask, cv::Rect{borderSize, borderSize, imgWidth - 2 * borderSize, imgHeight - 2 * borderSize}, whiteColor, -1);

    std::sort(points.begin(), points.end(), [](const PointInfo &left, const PointInfo &right) { return left.cnt > right.cnt; });

    std::vector<PointInfo> maskPoints;
    for (auto &pt : points)
    {
        if (mask.at<uchar>(pt.p2D.y(), pt.p2D.x()) == 255)
        {
            // draw mask with black
            cv::circle(mask, cv::Point2d{pt.p2D.x(), pt.p2D.y()}, pointExtractMinDist, blackColor, -1);
            maskPoints.emplace_back(pt);
        }
    }

    frame->points = maskPoints;

    return mask;
}

void PointTracker::extractPoints(const Frame::Ptr &prev, const Frame::Ptr &curr)
{
    utility::TicToc td;

    auto &img = curr->leftImg;
    std::vector<cv::Point2f> newPoints;

    cv::Mat mask = this->sortPointsByCnt(curr);
    int needPointNum = pointExtractMaxNum - curr->points.size();
    if (needPointNum > 0)
    {
        CHECK_EQ(mask.type(), CV_8UC1);
        // LOG(INFO) << "needPointNum: " << needPointNum;

        if (pointExtractType == PointExtractFlag::FAST)
        {
            std::vector<cv::KeyPoint> kps, prevKps;
            for (auto &pt : curr->points)
            {
                cv::KeyPoint kp;
                kp.pt.x = pt.p2D.x();
                kp.pt.y = pt.p2D.y();
                prevKps.push_back(kp);
            }
            fast->setMaxFeature(needPointNum);
            // fast->detect(img, kps, mask);
            fast->detect(img, kps, mask, prevKps);
            cv::KeyPoint::convert(kps, newPoints);
        }
        else if (pointExtractType == PointExtractFlag::Harris)
        {
            std::vector<cv::KeyPoint> kps;
            harris->setMaxFeature(needPointNum);
            harris->detect(img, kps, mask);
            cv::KeyPoint::convert(kps, newPoints);
        }
        else if (pointExtractType == PointExtractFlag::None)
        {
            utility::TicToc dt;

            cv::goodFeaturesToTrack(img, newPoints, needPointNum, 0.01, pointExtractMinDist, mask);

#ifdef TEST_PERF
            {
                static double sum_dt = 0;
                static int cnt = 0;
                sum_dt += dt.toc();
                cnt++;
                printf("avg dt cost: %f\n", sum_dt / cnt);
            }
#endif
        }
        else
            LOG(ERROR) << "pointExtractType error";
    }
    else
        newPoints.clear();

    // deal with new pts
    for (auto &pt : newPoints)
    {
        PointInfo point;
        point.id = Frame::pointIDCnt++;
        point.cnt = 1;
        point.p2D = Eigen::Vector2d{pt.x, pt.y};
        curr->points.emplace_back(point);
    }

#ifdef TEST_PERF
    {
        static double sum_dt = 0;
        static int cnt = 0;
        sum_dt += td.toc();
        cnt++;
        printf("extract point cost: %f\n", sum_dt / cnt);
    }

    {
        static double err = 0;
        static int cnt = 0;

        if (!newPoints.empty())
        {
            auto refinePoints = newPoints;
            cv::cornerSubPix(img, refinePoints, cv::Size{5, 5}, cv::Size{-1, -1},
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001));

            for (int i = 0; i < refinePoints.size(); i++)
            {
                double error = cv::norm(refinePoints[i] - newPoints[i]);
                err += error;
                cnt++;
            }
            printf("extract point err: %f\n", err / cnt);
        }
    }
#endif

    LOG_EVERY_N(INFO, logFreq) << "extract " << newPoints.size() << " points cost: " << td.toc() << "ms";
}

void PointTracker::rejectWithF(const Frame::Ptr &prev, const Frame::Ptr &curr)
{
    if (!prev || !curr)
        return;

    auto &currPts = curr->points;
    auto &prevPts = prev->points;

    if (currPts.size() >= 8)
    {
        utility::TicToc td;

        std::vector<uchar> status;
        std::vector<cv::Point2d> unCurrPts(currPts.size()), unPrevPts(currPts.size());
        for (unsigned int i = 0; i < currPts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            camera->liftProjective(currPts[i].p2D, tmp_p);
            tmp_p.x() = focalLength * tmp_p.x() / tmp_p.z() + imgWidth / 2.0;
            tmp_p.y() = focalLength * tmp_p.y() / tmp_p.z() + imgHeight / 2.0;
            unCurrPts[i] = cv::Point2d(tmp_p.x(), tmp_p.y());

            camera->liftProjective(prevPts[i].p2D, tmp_p);
            tmp_p.x() = focalLength * tmp_p.x() / tmp_p.z() + imgWidth / 2.0;
            tmp_p.y() = focalLength * tmp_p.y() / tmp_p.z() + imgHeight / 2.0;
            unPrevPts[i] = cv::Point2d(tmp_p.x(), tmp_p.y());
        }

        cv::findFundamentalMat(unCurrPts, unPrevPts, cv::FM_RANSAC, pointMatchRansacTh, 0.999, status);  // 1 pixel threshold

        // reduceVector with prev and curr frame
        {
            using namespace utility;
            reduceVector(prev->points, status);
            reduceVector(curr->points, status);
        }

#ifdef TEST_PERF
        {
            static int succ = 0, total = 0;
            succ += curr->points.size();
            total += status.size();
            printf("opticalflow track acc: %f\n", 1.f * succ / total);
        }
#endif

        // LOG_EVERY_N(INFO, logFreq) << "FM ransac costs: " << td.toc() << "ms";
    }
    else
        LOG(WARNING) << "PointMatcher rejectWithF input points less than 8, ransac filter outliers failed";
}

void PointTracker::matchByLKOpticFlow(const Frame::Ptr &prev, const Frame::Ptr &curr)
{
    if (!prev || prev->points.empty())
        return;

    utility::TicToc td;

    auto &prevImg = prev->leftImg;
    auto &currImg = curr->leftImg;
    std::vector<cv::Point2f> prevPoints, currPoints;
    for (auto &pt : prev->points)
        prevPoints.emplace_back(cv::Point2f{(float)pt.p2D.x(), (float)pt.p2D.y()});

    CHECK(!prevImg.empty());
    CHECK(!currImg.empty());

    std::vector<uchar> status(prevPoints.size());
    if (hasPredict)
    {
        std::vector<cv::Point2f> backwardPoints;
        std::vector<uchar> forwardStatus;
        std::vector<uchar> backwardStatus;

        // try lk optic flow by predict points
        currPoints = predictPoints;

        cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPoints, currPoints, forwardStatus, cv::noArray(), cv::Size(21, 21), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, precision), cv::OPTFLOW_USE_INITIAL_FLOW);

        // check forward optic flow with predict points
        int succNum = 0;
        for (size_t i = 0; i < forwardStatus.size(); i++)
        {
            if (forwardStatus[i])
                succNum++;
        }
        if (succNum < 10)
        {
            LOG(INFO) << "initial optic flow failed";
            cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPoints, currPoints, forwardStatus, cv::noArray(), cv::Size(21, 21), 3);
        }

        // backward optic flow
        backwardPoints = prevPoints;
        cv::calcOpticalFlowPyrLK(currImg, prevImg, currPoints, backwardPoints, backwardStatus, cv::noArray(), cv::Size(21, 21), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, precision), cv::OPTFLOW_USE_INITIAL_FLOW);
        for (unsigned i = 0; i < prevPoints.size(); i++)
        {
            if (forwardStatus[i] && backwardStatus[i] && cv::norm(prevPoints[i] - backwardPoints[i]) <= 0.5)
                status[i] = 1;
            else
                status[i] = 0;
        }

        hasPredict = false;
    }
    else
    {
        std::vector<cv::Point2f> backwardPoints;
        std::vector<uchar> forwardStatus;
        std::vector<uchar> backwardStatus;

        cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPoints, currPoints, forwardStatus, cv::noArray(), cv::Size(21, 21), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, precision));
        backwardPoints = prevPoints;
        cv::calcOpticalFlowPyrLK(currImg, prevImg, currPoints, backwardPoints, backwardStatus, cv::noArray(), cv::Size(21, 21), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, precision), cv::OPTFLOW_USE_INITIAL_FLOW);
        for (unsigned i = 0; i < prevPoints.size(); i++)
        {
            if (forwardStatus[i] && backwardStatus[i] && cv::norm(prevPoints[i] - backwardPoints[i]) <= 0.5)
                status[i] = 1;
            else
                status[i] = 0;
        }
    }

    // reduceVector with prev and curr frame
    for (unsigned i = 0; i < prev->points.size(); i++)
    {
        if (status[i])
        {
            PointInfo point;
            point.id = prev->points[i].id;
            point.cnt = prev->points[i].cnt + 1;
            point.p2D = Eigen::Vector2d{currPoints[i].x, currPoints[i].y};

            curr->points.emplace_back(point);
        }
    }
    utility::reduceVector(prev->points, status);

    LOG_EVERY_N(INFO, logFreq) << "point optic flow match cost: " << td.toc() << "ms";

#ifdef TEST_PERF
    {
        static double sum_dt = 0;
        static int cnt = 0;
        sum_dt += td.toc();
        cnt++;
        printf("optical flow point cost: %f\n", sum_dt / cnt);
    }

    {
        static double err = 0;
        static int cnt = 0;

        if (!currPoints.empty())
        {
            auto refinePoints = currPoints;
            cv::cornerSubPix(currImg, refinePoints, cv::Size{5, 5}, cv::Size{-1, -1},
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001));

            for (int i = 0; i < refinePoints.size(); i++)
            {
                double error = cv::norm(refinePoints[i] - currPoints[i]);
                err += error;
                cnt++;
            }
            printf("optical flow err: %f\n", err / cnt);
        }
    }
#endif
}

void PointTracker::matchByFBOpticFlow(const Frame::Ptr &prev, const Frame::Ptr &curr)
{
    if (!prev || prev->points.empty())
        return;

    utility::TicToc td;

    auto &prevImg = prev->leftImg;
    auto &currImg = curr->leftImg;
    std::vector<cv::Point2f> prevPoints, currPoints;
    for (auto &pt : prev->points)
        prevPoints.emplace_back(cv::Point2f{(float)pt.p2D.x(), (float)pt.p2D.y()});

    CHECK(!prevImg.empty());
    CHECK(!currImg.empty());

    std::vector<uchar> status;
    {
        cv::Mat forwardFlow, backwardFlow;
        // cv::optflow::calcOpticalFlowSparseToDense(prevImg, currImg, forwardFlow);
        // cv::optflow::calcOpticalFlowSparseToDense(currImg, prevImg, backwardFlow);
        forwardFlow = curr->p2cImg;

        for (auto &prevPt : prevPoints)
        {
            cv::Vec2f flow = forwardFlow.at<cv::Vec2f>(prevPt);
            cv::Point2f currPt = prevPt + cv::Point2f{flow[0], flow[1]};
            currPoints.push_back(currPt);

            if (inBorder(currPt))
                status.push_back(1);
            else
                status.push_back(0);
        }
    }

    // if (hasPredict)
    // {
    //     std::vector<cv::Point2f> backwardPoints;
    //     std::vector<uchar> forwardStatus;
    //     std::vector<uchar> backwardStatus;

    //     // try lk optic flow by predict points
    //     currPoints = predictPoints;

    //     cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPoints, currPoints, forwardStatus, cv::noArray(), cv::Size(21, 21), 1,
    //                              cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

    //     // check forward optic flow with predict points
    //     int succNum = 0;
    //     for (size_t i = 0; i < forwardStatus.size(); i++)
    //     {
    //         if (forwardStatus[i])
    //             succNum++;
    //     }
    //     if (succNum < 10)
    //     {
    //         LOG(INFO) << "initial optic flow failed";
    //         cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPoints, currPoints, forwardStatus, cv::noArray(), cv::Size(21, 21), 3);
    //     }

    //     // backward optic flow
    //     backwardPoints = prevPoints;
    //     cv::calcOpticalFlowPyrLK(currImg, prevImg, currPoints, backwardPoints, backwardStatus, cv::noArray(), cv::Size(21, 21), 1,
    //                              cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    //     for (unsigned i = 0; i < prevPoints.size(); i++)
    //     {
    //         if (forwardStatus[i] && backwardStatus[i] && cv::norm(prevPoints[i] - backwardPoints[i]) <= 0.5)
    //             status[i] = 1;
    //         else
    //             status[i] = 0;
    //     }

    //     hasPredict = false;
    // }
    // else
    // {
    //     std::vector<cv::Point2f> backwardPoints;
    //     std::vector<uchar> forwardStatus;
    //     std::vector<uchar> backwardStatus;

    //     cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPoints, currPoints, forwardStatus, cv::noArray(), cv::Size(21, 21), 3);
    //     backwardPoints = prevPoints;
    //     cv::calcOpticalFlowPyrLK(currImg, prevImg, currPoints, backwardPoints, backwardStatus, cv::noArray(), cv::Size(21, 21), 1,
    //                              cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    //     for (unsigned i = 0; i < prevPoints.size(); i++)
    //     {
    //         if (forwardStatus[i] && backwardStatus[i] && cv::norm(prevPoints[i] - backwardPoints[i]) <= 0.5)
    //             status[i] = 1;
    //         else
    //             status[i] = 0;
    //     }
    // }

    // reduceVector with prev and curr frame
    for (unsigned i = 0; i < prev->points.size(); i++)
    {
        if (status[i])
        {
            PointInfo point;
            point.id = prev->points[i].id;
            point.cnt = prev->points[i].cnt + 1;
            point.p2D = Eigen::Vector2d{currPoints[i].x, currPoints[i].y};

            curr->points.emplace_back(point);
        }
    }
    utility::reduceVector(prev->points, status);

    LOG_EVERY_N(INFO, logFreq) << "point optic flow match cost: " << td.toc() << "ms";

#ifdef TEST_PERF
    {
        static double sum_dt = 0;
        static int cnt = 0;
        sum_dt += td.toc();
        cnt++;
        printf("optical flow point cost: %f\n", sum_dt / cnt);
    }

    {
        static double err = 0;
        static int cnt = 0;

        if (!currPoints.empty())
        {
            auto refinePoints = currPoints;
            cv::cornerSubPix(currImg, refinePoints, cv::Size{5, 5}, cv::Size{-1, -1},
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001));

            for (int i = 0; i < refinePoints.size(); i++)
            {
                double error = cv::norm(refinePoints[i] - currPoints[i]);
                err += error;
                cnt++;
            }
            printf("optical flow err: %f\n", err / cnt);
        }
    }
#endif
}

void PointTracker::matchRightPointsByLKOpticFlow(const Frame::Ptr &frame)
{
    if (!frame || frame->points.empty())
        return;

    utility::TicToc td;

    auto &leftImg = frame->leftImg;
    auto &rightImg = frame->rightImg;
    std::vector<cv::Point2f> currPoints, rightPoints, backwardPoints;
    for (auto &pt : frame->points)
        currPoints.emplace_back(cv::Point2f{(float)pt.p2D.x(), (float)pt.p2D.y()});

    CHECK(!leftImg.empty());
    CHECK(!rightImg.empty());

    std::vector<uchar> forwardStatus, backwardStatus;

    cv::calcOpticalFlowPyrLK(leftImg, rightImg, currPoints, rightPoints, forwardStatus, cv::noArray(), cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, precision));
    backwardPoints = currPoints;
    cv::calcOpticalFlowPyrLK(rightImg, leftImg, rightPoints, backwardPoints, backwardStatus, cv::noArray(), cv::Size(21, 21), 1,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, precision), cv::OPTFLOW_USE_INITIAL_FLOW);
    for (unsigned i = 0; i < forwardStatus.size(); i++)
    {
        if (forwardStatus[i] && backwardStatus[i] && inBorder(rightPoints[i]) && cv::norm(currPoints[i] - backwardPoints[i]) <= 0.5)
            forwardStatus[i] = 1;
        else
            forwardStatus[i] = 0;
    }

    for (unsigned i = 0; i < forwardStatus.size(); i++)
    {
        if (forwardStatus[i])
        {
            frame->points[i].stereo = true;
            frame->points[i].p2DRight = Eigen::Vector2d{rightPoints[i].x, rightPoints[i].y};
        }
    }

#ifdef TEST_PERF
    {
        static double sum_dt = 0;
        static int cnt = 0;
        sum_dt += td.toc();
        cnt++;
        printf("optical flow stereo point cost: %f\n", sum_dt / cnt);
    }

    {
        static double err = 0;
        static int cnt = 0;

        if (!rightPoints.empty())
        {
            auto refinePoints = rightPoints;
            cv::cornerSubPix(rightImg, refinePoints, cv::Size{5, 5}, cv::Size{-1, -1},
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001));

            for (int i = 0; i < refinePoints.size(); i++)
            {
                double error = cv::norm(refinePoints[i] - rightPoints[i]);
                err += error;
                cnt++;
            }
            printf("optical flow stereo err: %f\n", err / cnt);
        }
    }
#endif
}

void PointTracker::matchRightPointsByFBOpticFlow(const Frame::Ptr &frame)
{
    if (!frame || frame->points.empty())
        return;

    utility::TicToc td;

    auto &leftImg = frame->leftImg;
    auto &rightImg = frame->rightImg;
    std::vector<cv::Point2f> currPoints, rightPoints, backwardPoints;
    for (auto &pt : frame->points)
        currPoints.emplace_back(cv::Point2f{(float)pt.p2D.x(), (float)pt.p2D.y()});

    CHECK(!leftImg.empty());
    CHECK(!rightImg.empty());

    std::vector<uchar> status;
    {
        cv::Mat forwardFlow, backwardFlow;
        // cv::optflow::calcOpticalFlowSparseToDense(leftImg, rightImg, forwardFlow);
        // cv::optflow::calcOpticalFlowSparseToDense(currImg, prevImg, backwardFlow);
        forwardFlow = frame->l2rImg;

        for (auto &leftPt : currPoints)
        {
            cv::Vec2f flow = forwardFlow.at<cv::Vec2f>(leftPt);
            cv::Point2f rightPt = leftPt + cv::Point2f{flow[0], flow[1]};
            rightPoints.push_back(rightPt);

            if (inBorder(rightPt))
                status.push_back(1);
            else
                status.push_back(0);
        }
    }

    for (unsigned i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            frame->points[i].stereo = true;
            frame->points[i].p2DRight = Eigen::Vector2d{rightPoints[i].x, rightPoints[i].y};
        }
    }

#ifdef TEST_PERF
    {
        static double sum_dt = 0;
        static int cnt = 0;
        sum_dt += td.toc();
        cnt++;
        printf("optical flow stereo point cost: %f\n", sum_dt / cnt);
    }

    {
        static double err = 0;
        static int cnt = 0;

        if (!rightPoints.empty())
        {
            auto refinePoints = rightPoints;
            cv::cornerSubPix(rightImg, refinePoints, cv::Size{5, 5}, cv::Size{-1, -1},
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001));

            for (int i = 0; i < refinePoints.size(); i++)
            {
                double error = cv::norm(refinePoints[i] - rightPoints[i]);
                err += error;
                cnt++;
            }
            printf("optical flow stereo err: %f\n", err / cnt);
        }
    }
#endif
}
