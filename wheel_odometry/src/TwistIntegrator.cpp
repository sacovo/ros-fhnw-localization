
#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

class Integrator
{
public:
    Integrator(rclcpp::Node::SharedPtr node): node(node)
    {
        node->declare_parameter("twist_topic", "/wheel/twist");
        node->declare_parameter("pose_topic", "/wheel/pose");
        node->declare_parameter("path_topic", "/wheel/path");

        twist_sub = node->create_subscription<geometry_msgs::msg::TwistStamped>(
            node->get_parameter("twist_topic").as_string(),
            rclcpp::QoS(100),
            std::bind(&Integrator::twist_callback, this, std::placeholders::_1));

        pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>(
            node->get_parameter("pose_topic").as_string(), rclcpp::QoS(10));

        path_pub = node->create_publisher<nav_msgs::msg::Path>(
            node->get_parameter("path_topic").as_string(), rclcpp::QoS(10));
        tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(node);

        // Initialize position and orientation
        x = 0.0;
        y = 0.0;
        z = 0.0;
        yaw = 0.0;
        t = 0.0;
        init = false;

        // Initialize path message
        path_msg.header.frame_id = "global";
    }

    void twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg) 
    {
        double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        if (!init) {
            t = current_time;
            init = true;
            return;
        }

        double dt = current_time - t;
        t = current_time;

        // Perform Runge-Kutta integration
        runge_kutta_integration(msg->twist, dt);

        // Create and update the pose message
        auto pose_msg = geometry_msgs::msg::PoseStamped();
        pose_msg.header.stamp = msg->header.stamp;
        pose_msg.header.frame_id = "global"; // Set your frame id

        pose_msg.pose.position.x = x;
        pose_msg.pose.position.y = y;
        pose_msg.pose.position.z = z;

        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();

        RCLCPP_INFO(node->get_logger(), "x: %f, y: %f, yaw: %f", x, y, yaw);

        // Publish the updated pose
        pose_pub->publish(pose_msg);

        
        // Add the pose to the path and publish it
        if (path_msg.poses.empty() || distance(path_msg.poses.back(), pose_msg) > 0.1) {
            
            // Add the pose to the path and publish it
            path_msg.header.stamp = msg->header.stamp;
            path_msg.poses.push_back(pose_msg);
            path_pub->publish(path_msg);
        }
        // Publish the transform
        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = msg->header.stamp;
        transform_stamped.header.frame_id = "global";
        transform_stamped.child_frame_id = "wheel";

        transform_stamped.transform.translation.x = x;
        transform_stamped.transform.translation.y = y;
        transform_stamped.transform.translation.z = z;
        transform_stamped.transform.rotation = pose_msg.pose.orientation;

        tf_broadcaster->sendTransform(transform_stamped);
    }


private:
    
    
    void runge_kutta_integration(const geometry_msgs::msg::Twist &twist, double dt)
    {
        // k1 terms
        double k1_x = twist.linear.x * cos(yaw) - twist.linear.y * sin(yaw);
        double k1_y = twist.linear.x * sin(yaw) + twist.linear.y * cos(yaw);
        double k1_yaw = twist.angular.z;

        // Intermediate terms
        double x_mid = x + 0.5 * k1_x * dt;
        double y_mid = y + 0.5 * k1_y * dt;
        double yaw_mid = yaw + 0.5 * k1_yaw * dt;

        // k2 terms
        double k2_x = twist.linear.x * cos(yaw_mid) - twist.linear.y * sin(yaw_mid);
        double k2_y = twist.linear.x * sin(yaw_mid) + twist.linear.y * cos(yaw_mid);
        double k2_yaw = twist.angular.z;

        // Intermediate terms
        x_mid = x + 0.5 * k2_x * dt;
        y_mid = y + 0.5 * k2_y * dt;
        yaw_mid = yaw + 0.5 * k2_yaw * dt;

        // k3 terms
        double k3_x = twist.linear.x * cos(yaw_mid) - twist.linear.y * sin(yaw_mid);
        double k3_y = twist.linear.x * sin(yaw_mid) + twist.linear.y * cos(yaw_mid);
        double k3_yaw = twist.angular.z;

        // Intermediate terms
        double x_end = x + k3_x * dt;
        double y_end = y + k3_y * dt;
        double yaw_end = yaw + k3_yaw * dt;

        // k4 terms
        double k4_x = twist.linear.x * cos(yaw_end) - twist.linear.y * sin(yaw_end);
        double k4_y = twist.linear.x * sin(yaw_end) + twist.linear.y * cos(yaw_end);
        double k4_yaw = twist.angular.z;

        // Update position and orientation using RK4 formula
        x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
        y += (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y);
        yaw += (dt / 6.0) * (k1_yaw + 2.0 * k2_yaw + 2.0 * k3_yaw + k4_yaw);
    }

    double distance(const geometry_msgs::msg::PoseStamped &pose1, const geometry_msgs::msg::PoseStamped &pose2) {
        return std::sqrt(std::pow(pose1.pose.position.x - pose2.pose.position.x, 2) +
                         std::pow(pose1.pose.position.y - pose2.pose.position.y, 2) +
                         std::pow(pose1.pose.position.z - pose2.pose.position.z, 2));
    }

    double x, y, z;
    double yaw;
    double t;
    bool init;

    rclcpp::Node::SharedPtr node;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_sub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    nav_msgs::msg::Path path_msg;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    auto node = std::make_shared<rclcpp::Node>("twist_integrator", options);
    auto calc = Integrator(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}