#include "utils/Visualization.h"

Visualization::Visualization() : queueSize(10)
{
    LOG(INFO) << "Visualization create";

    // worker
    visualCam = std::make_shared<CameraPoseVisualization>(1, 0, 0, 1);
    visualCam->setScale(0.5);
    visualCam->setLineWidth(0.05);

    // visual pose
    visualPose.type = visualization_msgs::Marker::SPHERE_LIST;
    visualPose.action = visualization_msgs::Marker::ADD;
    visualPose.pose.orientation.w = 1;
    visualPose.lifetime = ros::Duration();
    visualPose.scale.x = 0.1;
    visualPose.scale.y = 0.1;
    visualPose.scale.z = 0.1;
    visualPose.color.a = 1.0;
    visualPose.color.b = 0.0;
    visualPose.color.g = 1.0;
    visualPose.color.r = 1.0;
    visualPose.id = 0;
}

void Visualization::registerPub(ros::NodeHandle &n)
{
    LOG(INFO) << "Visualization registerPub";

    // pose relative
    pubOdometry = n.advertise<nav_msgs::Odometry>("odometry", 1);
    pubPath = n.advertise<nav_msgs::Path>("path", 1);
    pubVisualPose = n.advertise<visualization_msgs::Marker>("visual_pose", 1);
    pubVisualCam = n.advertise<visualization_msgs::MarkerArray>("visual_cam", 1000);

    // viusal img relativa
    pubPointExtract = n.advertise<sensor_msgs::Image>("point_extract", 1);

    // spatial relative
    pubPointCloud = n.advertise<sensor_msgs::PointCloud2>("pointclouds", 1);
    pubSpatialPoint = n.advertise<sensor_msgs::PointCloud>("spatial_point", 1);
    pubMarginPoint = n.advertise<sensor_msgs::PointCloud>("margin_point", 1);

    // loop
    pubKeyPose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1);
    pubExtrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1);
    pubKeyPoint = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1);
}

void Visualization::inputImg(const VisualizationImg &visual)
{
    std_msgs::Header header;
    header.seq = visual.frameID;
    header.frame_id = "camera";
    header.stamp = ros::Time(visual.timeStamp);

    if (!visual.depthImg.empty())
    {
        auto pointClouds = utility::getPointCloudFromMat(visual.depthImg, params.K, params.depthTh);
        this->publishPointCloud(VisualizationFlag::PointClouds, header, pointClouds);
    }

    if (!visual.pointImg.empty())
        this->publishImg(VisualizationFlag::PointExtract, header, visual.pointImg);
}

void Visualization::inputFeature(const Visualization3D &visual)
{
    std_msgs::Header header;
    header.seq = visual.frameID;
    header.frame_id = "world";
    header.stamp = ros::Time(visual.timeStamp);

    if (!visual.pts3D.empty())
        this->publishSpatilaPoint(VisualizationFlag::SpatialPoint, header, visual.pts3D);

    if (!visual.marginPts3D.empty())
        this->publishMarginPoint(VisualizationFlag::MarginPoint, header, visual.marginPts3D);

    // if (!visual.marginLines3D.empty())
    // {
    //     for (auto &idLine : visual.marginLines3D)
    //     {
    //         auto &id = idLine.first;
    //         auto &line = idLine.second;
    //         auto iter = marginlines.find(id);
    //         if (iter != marginlines.end())
    //         {
    //             iter->second = line;
    //         }
    //         else
    //             marginlines.insert(std::make_pair(id, line));
    //     }
    //     this->publishMarginLine(VisualizationFlag::MarginLine, header);
    // }
}

void Visualization::inputOdom(const VisualizationOdom &visual)
{
    std_msgs::Header header;
    header.seq = visual.frameID;
    header.frame_id = "world";
    header.stamp = ros::Time(visual.timeStamp);

    this->publishOdometry(VisualizationFlag::Odometry, header, "body", visual.P, visual.V, visual.R);
    this->publishTF(visual);

    std::vector<Eigen::Quaterniond> Rcam;
    std::vector<Eigen::Vector3d> tcam;
    for (unsigned i = 0; i < visual.Ric.size(); i++)
    {
        Rcam.emplace_back(Eigen::Quaterniond{visual.R * visual.Ric[i]});
        tcam.emplace_back(visual.P + visual.R.toRotationMatrix() * visual.tic[i]);
    }

    this->publishCamera(VisualizationFlag::Odometry, header, tcam, Rcam);
}

void Visualization::inputKeyframe(const Keyframe &kf)
{
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(kf.timeStamp);

    Eigen::Vector3d t = kf.t;
    Eigen::Quaterniond R{kf.R};

    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(kf.timeStamp);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = t.x();
    odometry.pose.pose.position.y = t.y();
    odometry.pose.pose.position.z = t.z();
    odometry.pose.pose.orientation.x = R.x();
    odometry.pose.pose.orientation.y = R.y();
    odometry.pose.pose.orientation.z = R.z();
    odometry.pose.pose.orientation.w = R.w();
    pubKeyPose.publish(odometry);

    Eigen::Vector3d tic = kf.tic;
    Eigen::Quaterniond Ric{kf.Ric};

    nav_msgs::Odometry extrinsic;
    extrinsic.header.stamp = ros::Time(kf.timeStamp);
    extrinsic.header.frame_id = "world";
    extrinsic.pose.pose.position.x = tic.x();
    extrinsic.pose.pose.position.y = tic.y();
    extrinsic.pose.pose.position.z = tic.z();
    extrinsic.pose.pose.orientation.x = Ric.x();
    extrinsic.pose.pose.orientation.y = Ric.y();
    extrinsic.pose.pose.orientation.z = Ric.z();
    extrinsic.pose.pose.orientation.w = Ric.w();
    pubExtrinsic.publish(extrinsic);

    sensor_msgs::PointCloud point_cloud;
    point_cloud.header.stamp = ros::Time(kf.timeStamp);
    point_cloud.header.frame_id = "world";
    for (int i = 0; i < kf.pts3D.size(); i++)
    {
        geometry_msgs::Point32 p;
        p.x = kf.pts3D[i](0);
        p.y = kf.pts3D[i](1);
        p.z = kf.pts3D[i](2);
        point_cloud.points.push_back(p);

        sensor_msgs::ChannelFloat32 p_2d;
        p_2d.values.push_back(kf.ptsUn[i].x());
        p_2d.values.push_back(kf.ptsUn[i].y());
        p_2d.values.push_back(kf.ptsUv[i].x());
        p_2d.values.push_back(kf.ptsUv[i].y());
        p_2d.values.push_back(kf.ptsID[i]);
        point_cloud.channels.push_back(p_2d);
    }

    pubKeyPoint.publish(point_cloud);
}

sensor_msgs::PointCloud2 Visualization::publishPointCloud(const VisualizationFlag &flag, const std_msgs::Header &header,
                                                          const pcl::PointCloud<pcl::PointXYZ>::Ptr &thisCloud)
{
    sensor_msgs::PointCloud2 pointCloudROS;
    if (flag != VisualizationFlag::PointClouds)
        return pointCloudROS;

    pcl::toROSMsg(*thisCloud, pointCloudROS);
    pointCloudROS.header = header;
    if (pubPointCloud.getNumSubscribers() != 0)
        pubPointCloud.publish(pointCloudROS);

    return pointCloudROS;
}

void Visualization::publishOdometry(const VisualizationFlag &flag, const std_msgs::Header &header, const std::string &frameID,
                                    const Eigen::Vector3d &P, const Eigen::Vector3d &V, const Eigen::Quaterniond &Q)
{
    if (flag != VisualizationFlag::Odometry)
        return;

    nav_msgs::Odometry odometryROS;
    odometryROS.header = header;
    odometryROS.child_frame_id = frameID;
    odometryROS.pose.pose.position.x = P.x();
    odometryROS.pose.pose.position.y = P.y();
    odometryROS.pose.pose.position.z = P.z();
    odometryROS.twist.twist.linear.x = V.x();
    odometryROS.twist.twist.linear.y = V.y();
    odometryROS.twist.twist.linear.z = V.z();
    odometryROS.pose.pose.orientation.w = Q.w();
    odometryROS.pose.pose.orientation.x = Q.x();
    odometryROS.pose.pose.orientation.y = Q.y();
    odometryROS.pose.pose.orientation.z = Q.z();
    if (pubOdometry.getNumSubscribers() != 0)
        pubOdometry.publish(odometryROS);

    // visualPose.header = header;
    // visualPose.points.push_back(odometryROS.pose.pose.position);
    // pubVisualPose.publish(visualPose);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometryROS.pose.pose;
    path.header = header;
    path.header.frame_id = "world";
    path.poses.push_back(pose_stamped);
    if (pubPath.getNumSubscribers() != 0)
        pubPath.publish(path);
}

void Visualization::publishCamera(const VisualizationFlag &flag, const std_msgs::Header &header, const std::vector<Eigen::Vector3d> &P,
                                  const std::vector<Eigen::Quaterniond> &Q)
{
    visualCam->reset();
    for (unsigned i = 0; i < P.size(); i++)
        visualCam->add_pose(P[i], Q[i]);
    visualCam->publish_by(pubVisualCam, header);
}

void Visualization::publishTF(const VisualizationOdom &visual)
{
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;

    // body frame
    transform.setOrigin(tf::Vector3(visual.P.x(), visual.P.y(), visual.P.z()));
    transform.setRotation(tf::Quaternion{visual.R.x(), visual.R.y(), visual.R.z(), visual.R.w()});
    br.sendTransform(tf::StampedTransform(transform, ros::Time(visual.timeStamp), "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(visual.tic[0].x(), visual.tic[0].y(), visual.tic[0].z()));
    transform.setRotation(tf::Quaternion{visual.Ric[0].x(), visual.Ric[0].y(), visual.Ric[0].z(), visual.Ric[0].w()});
    br.sendTransform(tf::StampedTransform(transform, ros::Time(visual.timeStamp), "body", "camera"));
}

void Visualization::publishImg(const VisualizationFlag &flag, const std_msgs::Header &header, const cv::Mat &showImg)
{
    sensor_msgs::ImagePtr imgROS = cv_bridge::CvImage(header, "bgr8", showImg).toImageMsg();
    switch (flag)
    {
    case VisualizationFlag::PointExtract:
        if (pubPointExtract.getNumSubscribers() != 0)
            pubPointExtract.publish(imgROS);
        break;
    default:
        break;
    }
}

void Visualization::publishSpatilaPoint(const VisualizationFlag &flag, const std_msgs::Header &header, const std::vector<Eigen::Vector3d> &points3D)
{
    if (flag != VisualizationFlag::SpatialPoint || points3D.empty())
        return;

    sensor_msgs::PointCloud pointClouds;
    pointClouds.header = header;
    for (auto &pt : points3D)
    {
        geometry_msgs::Point32 p3D;
        // auto point = utility::p3DFromCam(pt);
        // p3D.x = point.x();
        // p3D.y = point.y();
        // p3D.z = point.z();
        p3D.x = pt.x();
        p3D.y = pt.y();
        p3D.z = pt.z();
        pointClouds.points.push_back(p3D);
    }

    pubSpatialPoint.publish(pointClouds);
}

void Visualization::publishMarginPoint(const VisualizationFlag &flag, const std_msgs::Header &header, const std::vector<Eigen::Vector3d> &points3D)
{
    if (flag != VisualizationFlag::MarginPoint || points3D.empty())
        return;

    sensor_msgs::PointCloud marginPoints;
    marginPoints.header = header;
    for (auto &pt : points3D)
    {
        geometry_msgs::Point32 p3D;
        p3D.x = pt.x();
        p3D.y = pt.y();
        p3D.z = pt.z();
        marginPoints.points.push_back(p3D);
    }

    pubMarginPoint.publish(marginPoints);
}

// camera marker
const Eigen::Vector3d CameraPoseVisualization::imlt = Eigen::Vector3d(-1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::imrt = Eigen::Vector3d(1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::imlb = Eigen::Vector3d(-1.0, 0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::imrb = Eigen::Vector3d(1.0, 0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::lt0 = Eigen::Vector3d(-0.7, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::lt1 = Eigen::Vector3d(-0.7, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualization::lt2 = Eigen::Vector3d(-1.0, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualization::oc = Eigen::Vector3d(0.0, 0.0, 0.0);

CameraPoseVisualization::CameraPoseVisualization(float r, float g, float b, float a)
    : m_marker_ns("CameraPoseVisualization"), m_scale(0.2), m_line_width(0.01)
{
    m_image_boundary_color.r = r;
    m_image_boundary_color.g = g;
    m_image_boundary_color.b = b;
    m_image_boundary_color.a = a;
    m_optical_center_connector_color.r = r;
    m_optical_center_connector_color.g = g;
    m_optical_center_connector_color.b = b;
    m_optical_center_connector_color.a = a;
}

void CameraPoseVisualization::setImageBoundaryColor(float r, float g, float b, float a)
{
    m_image_boundary_color.r = r;
    m_image_boundary_color.g = g;
    m_image_boundary_color.b = b;
    m_image_boundary_color.a = a;
}

void CameraPoseVisualization::setOpticalCenterConnectorColor(float r, float g, float b, float a)
{
    m_optical_center_connector_color.r = r;
    m_optical_center_connector_color.g = g;
    m_optical_center_connector_color.b = b;
    m_optical_center_connector_color.a = a;
}

void CameraPoseVisualization::setScale(double s)
{
    m_scale = s;
}
void CameraPoseVisualization::setLineWidth(double width)
{
    m_line_width = width;
}
void CameraPoseVisualization::add_edge(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1)
{
    visualization_msgs::Marker marker;

    marker.ns = m_marker_ns;
    marker.id = m_markers.size() + 1;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.005;

    marker.color.g = 1.0f;
    marker.color.a = 1.0;

    geometry_msgs::Point point0, point1;

    utility::Eigen2Point(p0, point0);
    utility::Eigen2Point(p1, point1);

    marker.points.push_back(point0);
    marker.points.push_back(point1);

    m_markers.push_back(marker);
}

void CameraPoseVisualization::add_loopedge(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1)
{
    visualization_msgs::Marker marker;

    marker.ns = m_marker_ns;
    marker.id = m_markers.size() + 1;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.04;
    // marker.scale.x = 0.3;
    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;

    marker.color.r = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0;

    geometry_msgs::Point point0, point1;

    utility::Eigen2Point(p0, point0);
    utility::Eigen2Point(p1, point1);

    marker.points.push_back(point0);
    marker.points.push_back(point1);

    m_markers.push_back(marker);
}

void CameraPoseVisualization::add_pose(const Eigen::Vector3d &p, const Eigen::Quaterniond &q)
{
    visualization_msgs::Marker marker;

    marker.ns = m_marker_ns;
    marker.id = m_markers.size() + 1;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = m_line_width;

    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;

    geometry_msgs::Point pt_lt, pt_lb, pt_rt, pt_rb, pt_oc, pt_lt0, pt_lt1, pt_lt2;

    utility::Eigen2Point(q * (m_scale * imlt) + p, pt_lt);
    utility::Eigen2Point(q * (m_scale * imlb) + p, pt_lb);
    utility::Eigen2Point(q * (m_scale * imrt) + p, pt_rt);
    utility::Eigen2Point(q * (m_scale * imrb) + p, pt_rb);
    utility::Eigen2Point(q * (m_scale * lt0) + p, pt_lt0);
    utility::Eigen2Point(q * (m_scale * lt1) + p, pt_lt1);
    utility::Eigen2Point(q * (m_scale * lt2) + p, pt_lt2);
    utility::Eigen2Point(q * (m_scale * oc) + p, pt_oc);

    // image boundaries
    marker.points.push_back(pt_lt);
    marker.points.push_back(pt_lb);
    marker.colors.push_back(m_image_boundary_color);
    marker.colors.push_back(m_image_boundary_color);

    marker.points.push_back(pt_lb);
    marker.points.push_back(pt_rb);
    marker.colors.push_back(m_image_boundary_color);
    marker.colors.push_back(m_image_boundary_color);

    marker.points.push_back(pt_rb);
    marker.points.push_back(pt_rt);
    marker.colors.push_back(m_image_boundary_color);
    marker.colors.push_back(m_image_boundary_color);

    marker.points.push_back(pt_rt);
    marker.points.push_back(pt_lt);
    marker.colors.push_back(m_image_boundary_color);
    marker.colors.push_back(m_image_boundary_color);

    // top-left indicator
    marker.points.push_back(pt_lt0);
    marker.points.push_back(pt_lt1);
    marker.colors.push_back(m_image_boundary_color);
    marker.colors.push_back(m_image_boundary_color);

    marker.points.push_back(pt_lt1);
    marker.points.push_back(pt_lt2);
    marker.colors.push_back(m_image_boundary_color);
    marker.colors.push_back(m_image_boundary_color);

    // optical center connector
    marker.points.push_back(pt_lt);
    marker.points.push_back(pt_oc);
    marker.colors.push_back(m_optical_center_connector_color);
    marker.colors.push_back(m_optical_center_connector_color);

    marker.points.push_back(pt_lb);
    marker.points.push_back(pt_oc);
    marker.colors.push_back(m_optical_center_connector_color);
    marker.colors.push_back(m_optical_center_connector_color);

    marker.points.push_back(pt_rt);
    marker.points.push_back(pt_oc);
    marker.colors.push_back(m_optical_center_connector_color);
    marker.colors.push_back(m_optical_center_connector_color);

    marker.points.push_back(pt_rb);
    marker.points.push_back(pt_oc);
    marker.colors.push_back(m_optical_center_connector_color);
    marker.colors.push_back(m_optical_center_connector_color);

    m_markers.push_back(marker);
}

void CameraPoseVisualization::reset()
{
    m_markers.clear();
}

void CameraPoseVisualization::publish_by(ros::Publisher &pub, const std_msgs::Header &header)
{
    visualization_msgs::MarkerArray markerArray_msg;
    for (auto &marker : m_markers)
    {
        marker.header = header;
        markerArray_msg.markers.push_back(marker);
    }

    pub.publish(markerArray_msg);
}