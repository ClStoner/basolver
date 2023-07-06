#pragma once

#include "utils/Marco.h"
#include "utils/Utility.h"
#include "utils/Params.h"
#include "utils/MsgTransform.h"

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Point32.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

class CameraPoseVisualization
{
  public:
    POINTER_TYPEDEFS(CameraPoseVisualization);
    std::string m_marker_ns;

    CameraPoseVisualization(float r, float g, float b, float a);

    void setImageBoundaryColor(float r, float g, float b, float a = 1.0);
    void setOpticalCenterConnectorColor(float r, float g, float b, float a = 1.0);
    void setScale(double s);
    void setLineWidth(double width);

    void add_pose(const Eigen::Vector3d &p, const Eigen::Quaterniond &q);
    void reset();

    void publish_by(ros::Publisher &pub, const std_msgs::Header &header);
    void add_edge(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1);
    void add_loopedge(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1);

  private:
    std::vector<visualization_msgs::Marker> m_markers;
    std_msgs::ColorRGBA m_image_boundary_color;
    std_msgs::ColorRGBA m_optical_center_connector_color;
    double m_scale;
    double m_line_width;

    static const Eigen::Vector3d imlt;
    static const Eigen::Vector3d imlb;
    static const Eigen::Vector3d imrt;
    static const Eigen::Vector3d imrb;
    static const Eigen::Vector3d oc;
    static const Eigen::Vector3d lt0;
    static const Eigen::Vector3d lt1;
    static const Eigen::Vector3d lt2;
};

class Visualization
{
  public:
    POINTER_TYPEDEFS(Visualization);
    Visualization();

    void registerPub(ros::NodeHandle &n);

    void inputImg(const VisualizationImg &visual);

    void inputFeature(const Visualization3D &visual);

    void inputOdom(const VisualizationOdom &visual);

    void inputKeyframe(const Keyframe &kf);

  private:
    sensor_msgs::PointCloud2 publishPointCloud(const VisualizationFlag &flag, const std_msgs::Header &header,
                                               const pcl::PointCloud<pcl::PointXYZ>::Ptr &thisCloud);

    void publishOdometry(const VisualizationFlag &flag, const std_msgs::Header &header, const std::string &frameID, const Eigen::Vector3d &P,
                         const Eigen::Vector3d &V, const Eigen::Quaterniond &Q);

    void publishCamera(const VisualizationFlag &flag, const std_msgs::Header &header, const std::vector<Eigen::Vector3d> &P,
                       const std::vector<Eigen::Quaterniond> &Q);

    void publishTF(const VisualizationOdom &visual);

    void publishImg(const VisualizationFlag &flag, const std_msgs::Header &header, const cv::Mat &showImg);

    void publishSpatilaPoint(const VisualizationFlag &flag, const std_msgs::Header &header, const std::vector<Eigen::Vector3d> &points3D);

    void publishMarginPoint(const VisualizationFlag &flag, const std_msgs::Header &header, const std::vector<Eigen::Vector3d> &points3D);

    const int queueSize;

    ros::Publisher pubOdometry, pubPath, pubVisualPose, pubVisualCam;
    ros::Publisher pubPointExtract;
    ros::Publisher pubPointCloud;
    ros::Publisher pubSpatialPoint, pubMarginPoint, pubSpatilaLine, pubMarginLine;
    ros::Publisher pubKeyPose, pubKeyPoint, pubExtrinsic;                             //用于闭环

    // worker

    // params

    // variables
    nav_msgs::Path path;
    nav_msgs::Path loopPath;
    visualization_msgs::Marker visualPose;
    CameraPoseVisualization::Ptr visualCam;
};
