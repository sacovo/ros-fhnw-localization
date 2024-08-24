#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/int32.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <cv_bridge/cv_bridge.h>

// Struct for Camera
struct Camera {
    std::string name;
    Eigen::Matrix4d T_imu_cam;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::string topic;
    bool compressed;

    // Method for printing the Camera struct (to be implemented)
    friend std::ostream &operator<<(std::ostream &os, const Camera &camera);
};

// Struct for TagObservation
struct TagObservation {
    int marker_id;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    std::vector<cv::Point2f> corners;
    Camera cam;
};

// ArucoProcessor class definition
class ArucoProcessor {
public:
    // Constructor
    ArucoProcessor(
        const std::string &aruco_dict_name,
        double marker_length,
        const std::map<int, Eigen::Vector3d> &marker_positions,
        const std::vector<Camera> &camera_params
    );

    // Detect markers method (to be implemented)
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<int>> detectMarkers(cv::Mat &image);

    // Estimate tag pose method (to be implemented)
    std::pair<std::vector<TagObservation>, std::set<int>> estimateTagPose(
        const std::vector<std::vector<cv::Point2f>> &corners,
        const std::vector<int> &ids,
        Camera &cam,
        cv::Mat &image
    );

    // Reprojection error method (to be implemented)
    static double reprojectionError(
        const std::vector<double> &params,
        const std::vector<TagObservation> &observations,
        const std::map<int, Eigen::Vector3d> &marker_positions
    );

    // Project to image method (to be implemented)
    static std::vector<cv::Point2f> projectToImage(
        const Eigen::Vector3d &position,
        const double orientation[3],
        const Eigen::Vector3d &global_position,
        const cv::Vec3d &rvec,
        const Camera &cam
    );

    // Adjust tag position method (to be implemented)
    static Eigen::Vector3d adjustTagPosition(
        const Eigen::Vector3d &global_coords,
        const cv::Vec3d &rvec,
        double delta,
        const double orientation[3],
        const Eigen::Matrix3d &R_imu_cam
    );

    // Optimize pose method (to be implemented)
    std::pair<Eigen::Vector3d, Eigen::Vector3d> optimizePose(
        const Eigen::VectorXd &initial_guess,
        const std::vector<TagObservation> &observations
    );

private:
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    double marker_length_;
    std::map<int, Eigen::Vector3d> marker_positions_;
    std::vector<Camera> camera_params_;
};

// ArucoPoseEstimator class definition
class ArucoPoseEstimator : public rclcpp::Node {
public:
    ArucoPoseEstimator();

    // Odom callback method (to be implemented)
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

    // Get odom orientation method (to be implemented)
    Eigen::Vector3d getOdomOrientation();

    // Get odom position method (to be implemented)
    Eigen::Vector3d getOdomPosition();

    // Image callback method (to be implemented)
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    // Publish observations method (to be implemented)
    void publishObservations(const std::vector<TagObservation> &observations, const std_msgs::msg::Header &header);

    // Check callback frequency method (to be implemented)
    void checkCallbackFrequency();

    // Update pose method (to be implemented)
    void updatePose(const Eigen::Vector3d &position, const Eigen::Vector3d &orientation);

    // Query odom transform method (to be implemented)
    void queryOdomTransform();

    // Callback odom transform method (to be implemented)
    void callbackOdomTransform();

private:
    // Private member variables (to be defined in the constructor)
    std::map<int, Eigen::Vector3d> marker_positions_;
    std::vector<Camera> cameras_;
    std::map<int, rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> marker_pubs_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr count_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_odom_;
    rclcpp::Client<rcl_interfaces::srv::GetParameters>::SharedPtr client_;
    std::shared_ptr<cv_bridge::CvBridge> bridge_;
    std::shared_ptr<ArucoProcessor> processor_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    nav_msgs::msg::Odometry current_odom_;
    double yaw_;
    Eigen::Vector3d offset_position_;
    double last_callback_time_;
};
