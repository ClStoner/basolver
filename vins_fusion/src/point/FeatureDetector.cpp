#include "point/FeatureDetector.h"

OctTreeNode::OctTreeNode() : reproduce(true) {}

void OctTreeNode::divideNode(OctTreeNode &n1, OctTreeNode &n2, OctTreeNode &n3, OctTreeNode &n4)
{
    const int halfX = std::ceil(static_cast<float>(UR.x - UL.x) / 2);
    const int halfY = std::ceil(static_cast<float>(BR.y - UL.y) / 2);

    // Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x + halfX, UL.y);
    n1.BL = cv::Point2i(UL.x, UL.y + halfY);
    n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
    n1.kps.reserve(kps.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfY);
    n2.kps.reserve(kps.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.kps.reserve(kps.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.kps.reserve(kps.size());

    // Associate points to childs
    for (size_t i = 0; i < kps.size(); i++)
    {
        const cv::KeyPoint &kp = kps[i];
        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
                n1.kps.push_back(kp);
            else
                n3.kps.push_back(kp);
        }
        else if (kp.pt.y < n1.BR.y)
            n2.kps.push_back(kp);
        else
            n4.kps.push_back(kp);
    }

    if (n1.kps.size() == 1)
        n1.reproduce = true;
    if (n2.kps.size() == 1)
        n2.reproduce = true;
    if (n3.kps.size() == 1)
        n3.reproduce = true;
    if (n4.kps.size() == 1)
        n4.reproduce = true;
}

void OctTreeNode::drawNode(cv::Mat &_img)
{
    if (_img.channels() != 3)
        cv::cvtColor(_img, _img, cv::COLOR_GRAY2BGR);

    cv::line(_img, UR, UL, cv::Scalar{0, 0, 255}, 1);
    cv::line(_img, BR, BL, cv::Scalar{0, 0, 255}, 1);
    cv::line(_img, UR, BR, cv::Scalar{0, 0, 255}, 1);
    cv::line(_img, UL, BL, cv::Scalar{0, 0, 255}, 1);
}

FeatureDetector::FeatureDetector(int _max_feature, cv::Size _img_size, int _thereshold)
    : max_feature(_max_feature), thereshold(_thereshold), img_height(_img_size.height), img_width(_img_size.width)
{
    // LOG(INFO) << "FeatureDetector construct";
}

// TODO HarrisDetector
HarrisDetector::HarrisDetector(int _max_feature, cv::Size _img_size, int _thereshold, int _min_dist)
    : FeatureDetector(_max_feature, _img_size, _thereshold), min_dist(_min_dist)
{
    // LOG(INFO) << "FastDetector construct";

    harris = cv::GFTTDetector::create(max_feature, 0.01, min_dist);

    // LOG(INFO) << "grid_f_num: " << grid_f_num;
    // LOG(INFO) << "num_rows: " << num_rows << " num_cols: " << num_cols;
}

void HarrisDetector::drawKps()
{
    if (show_img.channels() != 3)
        cv::cvtColor(show_img, show_img, cv::COLOR_GRAY2BGR);

    for (auto &kp : keypoints)
        cv::circle(show_img, kp.pt, 1, cv::Scalar{0, 0, 255}, -1);

    // for (int i = 0; i <= num_cols; i++)
    //     cv::line(show_img, cv::Point2f{offset_X + cell_size * i, offset_Y}, cv::Point2f{offset_X + cell_size * i, img_height - offset_Y},
    //              cv::Scalar{255, 0, 0}, 1);

    // for (int i = 0; i <= num_rows; i++)
    //     cv::line(show_img, cv::Point2f{offset_X, offset_Y + cell_size * i}, cv::Point2f{img_width - offset_X, offset_Y + cell_size * i},
    //              cv::Scalar{255, 0, 0}, 1);

    static int cnt = 0;
    cv::imwrite("/root/share/myGit/mycode/intel/src/results/img/" + std::to_string(cnt++) + ".png", show_img);
    cv::imshow("OctTreeImg", show_img);
    cv::waitKey(10);
}

void HarrisDetector::clear()
{
    curr_img.release();
    show_img.release();
    mask.release();
    std::vector<cv::KeyPoint>().swap(keypoints);
}

void HarrisDetector::setMaxFeature(int _max_feature)
{
    max_feature = _max_feature;
    harris->setMaxFeatures(max_feature);
}

void HarrisDetector::detect(cv::Mat &_img, std::vector<cv::KeyPoint> &_kps, cv::Mat &_mask)
{
    LOG_IF(ERROR, _img.rows != img_height || _img.cols != img_width) << "input img size error";

    utility::TicToc dt;

    curr_img = _img.clone();
    show_img = _img.clone();
    if (_mask.empty())
        mask = cv::Mat(img_height, img_width, CV_8UC1, cv::Scalar(255));
    else
        mask = _mask.clone();

    // cv::goodFeaturesToTrack(curr_img, keypoints, max_feature, 0.01, min_dist, mask);
    harris->detect(curr_img, keypoints, mask);
    _kps = keypoints;

    // drawKps();
    clear();

    LOG(INFO) << "HarrisDetector detect " << _kps.size() << " cost: " << dt.toc();
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

// TODO FastDetector
FastDetector::FastDetector(int _max_feature, cv::Size _img_size, int _thereshold, int _cell_size)
    : FeatureDetector(_max_feature, _img_size, _thereshold), cell_size(_cell_size), num_rows(img_height / cell_size), num_cols(img_width / cell_size),
      offset_X((img_width - cell_size * num_cols) / 2), offset_Y((img_height - cell_size * num_rows) / 2),
      grid_f_num(2 * std::ceil(1.f * max_feature / (num_rows * num_cols)))
{
    // LOG(INFO) << "FastDetector construct";

    gridMask = std::vector<bool>(num_rows * num_cols, true);
    gridMaskCnt = std::vector<int>(num_rows * num_cols, 0);
    fast = cv::FastFeatureDetector::create(thereshold);
    // fast->setThreshold(thereshold);

    // LOG(INFO) << "grid_f_num: " << grid_f_num;
    // LOG(INFO) << "num_rows: " << num_rows << " num_cols: " << num_cols;
}

void FastDetector::detectGrid(cv::Point2i _grid_index)
{
    int col_index = _grid_index.x;
    int row_index = _grid_index.y;

    if (!gridMask[row_index * num_cols + col_index])
    {
        // LOG(INFO) << "miss";
        return;
    }

    int grid_start_x = offset_X + cell_size * col_index;
    int grid_end_x = offset_X + cell_size * (col_index + 1);
    int grid_start_y = offset_Y + cell_size * row_index;
    int grid_end_y = offset_Y + cell_size * (row_index + 1);

    // LOG(INFO) << "col_index: " << col_index << " row_index: " << row_index;
    // LOG(INFO) << "grid_start_x: " << grid_start_x << " grid_end_x: " << grid_end_x << " grid_start_y: " << grid_start_y
    //           << " grid_end_y: " << grid_end_y;
    cv::Mat grid_img = curr_img.rowRange(grid_start_y, grid_end_y).colRange(grid_start_x, grid_end_x);
    cv::Mat grid_mask = mask.rowRange(grid_start_y, grid_end_y).colRange(grid_start_x, grid_end_x);
    // LOG(INFO) << "col_index: " << col_index << " row_index: " << row_index << "get mat!";

    std::vector<cv::KeyPoint> kps;
    // cv::FAST(grid_img, kps, thereshold);
    fast->detect(grid_img, kps, grid_mask);

    // kps size less than grid feature size in need
    if (kps.size() < grid_f_num)
    {
        // LOG(INFO) << "low thereshold";
        kps.clear();
        // cv::FAST(grid_img, kps, thereshold / 2);
        fast->setThreshold(thereshold / 2);
        fast->detect(grid_img, kps, grid_mask);
        fast->setThreshold(thereshold);
    }

    if (kps.size() > grid_f_num)
    {
        std::sort(kps.begin(), kps.end(), [](cv::KeyPoint &left, cv::KeyPoint &right) { return left.response > right.response; });
        kps.resize(grid_f_num);
    }

    for (auto iter = kps.begin(); iter != kps.end(); iter++)
    {
        // std::unique_lock<std::mutex> lock(m_detect);
        iter->pt = iter->pt + cv::Point2f{grid_start_x, grid_start_y};
        keypoints.push_back(*iter);
    }

    // LOG(INFO) << "kps size: " << kps.size();
    // LOG(INFO) << "_kps size: " << keypoints.size();
}

void FastDetector::distriOctTree()
{
    // LOG(INFO) << "fast_detector distriOctTree begin";
    const int num_InitNode = std::round(static_cast<float>(img_width - 2 * offset_X) / (img_height - 2 * offset_Y));
    const float width_InitNode = static_cast<float>(img_width - 2 * offset_X) / num_InitNode;

    // init OcTtreeNodes location
    InitOctTreeNodes.resize(num_InitNode);
    for (int i = 0; i < num_InitNode; i++)
    {
        InitOctTreeNodes[i] = std::make_shared<OctTreeNode>();
        InitOctTreeNodes[i]->UL = cv::Point2i(offset_X + width_InitNode * static_cast<float>(i), offset_Y);
        InitOctTreeNodes[i]->UR = cv::Point2i(offset_X + width_InitNode * static_cast<float>(i + 1), offset_Y);
        InitOctTreeNodes[i]->BL = cv::Point2i(InitOctTreeNodes[i]->UL.x, img_height - offset_Y - offset_Y);
        InitOctTreeNodes[i]->BR = cv::Point2i(InitOctTreeNodes[i]->UR.x, img_height - offset_Y - offset_Y);
        OcTtreeNodes.push_back(InitOctTreeNodes[i]);
    }

    // init OcTtreeNodes kps
    for (auto &kp : keypoints)
        InitOctTreeNodes[(kp.pt.x - offset_X) / width_InitNode]->kps.push_back(kp);

    auto list_iter = OcTtreeNodes.begin();

    // set list reproduce flag
    while (list_iter != OcTtreeNodes.end())
    {
        if ((*list_iter)->kps.size() == 1)
        {
            (*list_iter)->reproduce = false;
            list_iter++;
        }
        else if ((*list_iter)->kps.empty())
            list_iter = OcTtreeNodes.erase(list_iter);
        else
            list_iter++;
    }

    int iteration = 0;
    bool finish_flag = false;
    std::vector<std::pair<int, OctTreeNode::Ptr>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(OcTtreeNodes.size() * 4);

    while (!finish_flag)
    {
        iteration++;
        int prevSize = OcTtreeNodes.size();
        int nToExpand = 0;
        list_iter = OcTtreeNodes.begin();
        vSizeAndPointerToNode.clear();

        // Traversal over OcTtreeNodes, divide them and erase them at last
        while (list_iter != OcTtreeNodes.end())
        {
            if (!(*list_iter)->reproduce)
            {
                // If node only contains one point do not subdivide and continue
                list_iter++;
                continue;
            }
            else
            {
                // LOG(INFO) << "reserve error" << iteration;

                // If more than one point, subdivide
                OctTreeNode n1, n2, n3, n4;
                (*list_iter)->divideNode(n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.kps.size() > 0)
                {
                    OcTtreeNodes.push_front(std::make_shared<OctTreeNode>(n1));
                    if (n1.kps.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n1.kps.size(), OcTtreeNodes.front()));
                        OcTtreeNodes.front()->lit = OcTtreeNodes.begin();
                    }
                }
                if (n2.kps.size() > 0)
                {
                    OcTtreeNodes.push_front(std::make_shared<OctTreeNode>(n2));
                    if (n2.kps.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n2.kps.size(), OcTtreeNodes.front()));
                        OcTtreeNodes.front()->lit = OcTtreeNodes.begin();
                    }
                }
                if (n3.kps.size() > 0)
                {
                    OcTtreeNodes.push_front(std::make_shared<OctTreeNode>(n3));
                    if (n3.kps.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n3.kps.size(), OcTtreeNodes.front()));
                        OcTtreeNodes.front()->lit = OcTtreeNodes.begin();
                    }
                }
                if (n4.kps.size() > 0)
                {
                    OcTtreeNodes.push_front(std::make_shared<OctTreeNode>(n4));
                    if (n4.kps.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n4.kps.size(), OcTtreeNodes.front()));
                        OcTtreeNodes.front()->lit = OcTtreeNodes.begin();
                    }
                }

                list_iter = OcTtreeNodes.erase(list_iter);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)OcTtreeNodes.size() >= max_feature || (int)OcTtreeNodes.size() == prevSize)
        {
            finish_flag = true;
        }
        else if (((int)OcTtreeNodes.size() + nToExpand * 3) > max_feature)
        {

            while (!finish_flag)
            {
                prevSize = OcTtreeNodes.size();

                std::vector<std::pair<int, OctTreeNode::Ptr>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                // Traversal nodes from those behave less kps, so sort nodes first and Traversal from backend
                std::sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    OctTreeNode n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->divideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.kps.size() > 0)
                    {
                        OcTtreeNodes.push_front(std::make_shared<OctTreeNode>(n1));
                        if (n1.kps.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n1.kps.size(), OcTtreeNodes.front()));
                            OcTtreeNodes.front()->lit = OcTtreeNodes.begin();
                        }
                    }
                    if (n2.kps.size() > 0)
                    {
                        OcTtreeNodes.push_front(std::make_shared<OctTreeNode>(n2));
                        if (n2.kps.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n2.kps.size(), OcTtreeNodes.front()));
                            OcTtreeNodes.front()->lit = OcTtreeNodes.begin();
                        }
                    }
                    if (n3.kps.size() > 0)
                    {
                        OcTtreeNodes.push_front(std::make_shared<OctTreeNode>(n3));
                        if (n3.kps.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n3.kps.size(), OcTtreeNodes.front()));
                            OcTtreeNodes.front()->lit = OcTtreeNodes.begin();
                        }
                    }
                    if (n4.kps.size() > 0)
                    {
                        OcTtreeNodes.push_front(std::make_shared<OctTreeNode>(n4));
                        if (n4.kps.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n4.kps.size(), OcTtreeNodes.front()));
                            OcTtreeNodes.front()->lit = OcTtreeNodes.begin();
                        }
                    }

                    OcTtreeNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int)OcTtreeNodes.size() >= max_feature)
                        break;
                }

                if ((int)OcTtreeNodes.size() >= max_feature || (int)OcTtreeNodes.size() == prevSize)
                    finish_flag = true;
            }
        }
    }

    // LOG(INFO) << "reserve error";
    keypoints.clear();
    keypoints.reserve(max_feature);
    int cnt = 0;
    for (std::list<OctTreeNode::Ptr>::iterator lit = OcTtreeNodes.begin(); lit != OcTtreeNodes.end(); lit++)
    {
        if (cnt >= max_feature)
            break;

        std::vector<cv::KeyPoint> &vNodeKeys = (*lit)->kps;
        cv::KeyPoint kp = *std::max_element(vNodeKeys.begin(), vNodeKeys.end(),
                                            [](const cv::KeyPoint &left, const cv::KeyPoint &right) { return left.response < right.response; });
        keypoints.push_back(kp);
        cnt++;
    }
    // LOG(INFO) << "fast_detector distriOctTree end!";
}

void FastDetector::updateMask(std::vector<cv::KeyPoint> &_kps)
{
    for (auto &kp : _kps)
    {
        int grid_row = (kp.pt.y - offset_Y) / cell_size;
        int grid_col = (kp.pt.x - offset_X) / cell_size;

        gridMaskCnt[grid_row * num_cols + grid_col]++;
        if (gridMaskCnt[grid_row * num_cols + grid_col] >= 1)
            gridMask[grid_row * num_cols + grid_col] = false;
    }
}

void FastDetector::drawKps()
{
    show_img = curr_img.clone();
    if (show_img.channels() != 3)
        cv::cvtColor(show_img, show_img, cv::COLOR_GRAY2BGR);

    for (auto &kp : keypoints)
    {
        if (kp.response < thereshold)
            cv::circle(show_img, kp.pt, 1, cv::Scalar{0, 255, 0}, -1);
        else
            cv::circle(show_img, kp.pt, 1, cv::Scalar{0, 0, 255}, -1);
    }

    for (auto &iter : OcTtreeNodes)
        iter->drawNode(show_img);

    // for (int i = 0; i <= num_cols; i++)
    //     cv::line(show_img, cv::Point2f{offset_X + cell_size * i, offset_Y}, cv::Point2f{offset_X + cell_size * i, img_height - offset_Y},
    //              cv::Scalar{255, 0, 0}, 1);

    // for (int i = 0; i <= num_rows; i++)
    //     cv::line(show_img, cv::Point2f{offset_X, offset_Y + cell_size * i}, cv::Point2f{img_width - offset_X, offset_Y + cell_size * i},
    //              cv::Scalar{255, 0, 0}, 1);

    static int cnt = 0;
    cv::imwrite("/root/share/myGit/mycode/intel/src/results/img/" + std::to_string(cnt++) + ".png", show_img);
    cv::imshow("OctTreeImg", show_img);
    cv::waitKey(10);
}

void FastDetector::setMaxFeature(int _max_feature)
{
    max_feature = _max_feature;
    grid_f_num = 2 * std::ceil(1.f * max_feature / (num_rows * num_cols));
}

void FastDetector::clear()
{
    std::vector<OctTreeNode::Ptr>().swap(InitOctTreeNodes);
    std::list<OctTreeNode::Ptr>().swap(OcTtreeNodes);
    std::vector<cv::KeyPoint>().swap(keypoints);

    gridMask = std::vector<bool>(num_rows * num_cols, true);
    gridMaskCnt = std::vector<int>(num_rows * num_cols, 0);
}

void FastDetector::detect(cv::Mat &_img, std::vector<cv::KeyPoint> &_kps, cv::Mat &_mask)
{
    LOG_IF(ERROR, _img.rows != img_height || _img.cols != img_width) << "input img size error";

    utility::TicToc dt;

    curr_img = _img.clone();
    show_img = _img.clone();
    if (_mask.empty())
        mask = cv::Mat(img_height, img_width, CV_8UC1, cv::Scalar(255));
    else
        mask = _mask;

    for (int i = 0; i < num_cols; i++)
    {
        for (int j = 0; j < num_rows; j++)
            detectGrid(cv::Point2i{i, j});
    }

    distriOctTree();
    // drawKps();
    _kps = keypoints;
    clear();

    LOG(INFO) << "fast_detector detect " << _kps.size() << " cost: " << dt.toc();

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

void FastDetector::detect(cv::Mat &_img, std::vector<cv::KeyPoint> &_kps, cv::Mat &_mask, std::vector<cv::KeyPoint> &_prev_kps)
{
    LOG_IF(ERROR, _img.rows != img_height || _img.cols != img_width) << "input img size error";

    utility::TicToc dt;

    curr_img = _img.clone();
    if (_mask.empty())
        mask = cv::Mat(img_height, img_width, CV_8UC1, cv::Scalar(255));
    else
        mask = _mask.clone();

    updateMask(_prev_kps);
    for (int i = 0; i < num_cols; i++)
    {
        for (int j = 0; j < num_rows; j++)
            detectGrid(cv::Point2i{i, j});
    }

    distriOctTree();
    // drawKps();
    _kps = keypoints;
    clear();

    // LOG(INFO) << "fast_detector detect " << _kps.size() << " cost: " << dt.toc();
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
