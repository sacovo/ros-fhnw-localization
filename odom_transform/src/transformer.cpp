#include "rclcpp/rclcpp.hpp"
#include "rclcpp/duration.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rcl_interfaces/msg/parameter_event.hpp"

class Transformer : public rclcpp::Node
{
public:
    Transformer() : Node("pose_transformer")
    {
        this->declare_parameter("yaw", 0.0);
        set_yaw(this->get_parameter("yaw"));

        this->declare_parameter("initial_orientation", std::vector<double>{0.0, 0.0, 0.0});
        set_initial_orientation(this->get_parameter("initial_orientation"));

        this->declare_parameter("initial_position", std::vector<double>{0.0, 0.0, 0.0});
        set_initial_position(this->get_parameter("initial_position"));

        // Setup listeners and publishers
        this->declare_parameter("odom_in", "");
        this->declare_parameter("odom_out", "");
        this->declare_parameter("pose_in", "");
        this->declare_parameter("pose_out", "");
        this->declare_parameter("path_in", "");
        this->declare_parameter("path_out", "");

        this->get_parameter("odom_in", odom_in);
        this->get_parameter("odom_out", odom_out);

        this->get_parameter("pose_out", pose_out);
        this->get_parameter("pose_in", pose_in);

        this->get_parameter("path_out", path_out);
        this->get_parameter("path_in", path_in);

        if (odom_in != "")
        {
            pub_odom = this->create_publisher<nav_msgs::msg::Odometry>(odom_out, 2);
            sub_odom = this->create_subscription<nav_msgs::msg::Odometry>(odom_in, 2, std::bind(&Transformer::callback_odom, this, std::placeholders::_1));
            RCLCPP_INFO(this->get_logger(), "Subscribed to %s, publishing to %s", odom_in.c_str(), odom_out.c_str());
        }

        if (pose_in != "")
        {
            pub_pose = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(pose_out, 2);
            sub_pose = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(pose_in, 2, std::bind(&Transformer::callback_pose, this, std::placeholders::_1));
            RCLCPP_INFO(this->get_logger(), "Subscribed to %s, publishing to %s", pose_in.c_str(), pose_out.c_str());
        }

        if (path_in != "")
        {
            pub_path = this->create_publisher<nav_msgs::msg::Path>(path_out, 2);
            sub_path = this->create_subscription<nav_msgs::msg::Path>(path_in, 2, std::bind(&Transformer::callback_path, this, std::placeholders::_1));
            RCLCPP_INFO(this->get_logger(), "Subscribed to %s, publishing to %s", path_in.c_str(), path_out.c_str());
        }
        parameter_subscription = rclcpp::AsyncParametersClient::on_parameter_event(
            this->get_node_topics_interface(),
            std::bind(&Transformer::on_parameter_changed_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Setup complete");
    }

    void on_parameter_changed_callback(std::shared_ptr<const rcl_interfaces::msg::ParameterEvent> event)
    {
        if (event->node != this->get_fully_qualified_name())
        {
            return;
        }

        for (const auto &param : event->changed_parameters)
        {
            if (param.name == "initial_position")
            {
                set_initial_position(this->get_parameter("initial_position"));
                RCLCPP_INFO(this->get_logger(), "Position was set: %.2f, %.2f, %.2f", set_position.getX(), set_position.getY(), set_position.getZ());
            }
            if (param.name == "initial_orientation")
            {
                set_initial_orientation(this->get_parameter("initial_orientation"));
                RCLCPP_INFO(this->get_logger(), "Orientation was set: %.2f, %.2f, %.2f, %.2f", set_orientation.getX(), set_orientation.getY(), set_orientation.getZ(), set_orientation.getZ());
            }
            if (param.name == "yaw")
            {
                set_yaw(this->get_parameter("yaw"));
                RCLCPP_INFO(this->get_logger(), "Yaw was set: %.2f", yaw);

                // Need to reset this as well, since it depends on our actual position
                // set_initial_position(this->get_parameter("initial_position"));
            }
        }
    }

    double deg2rad(double deg)
    {
        return M_PI * deg / 180.0;
    }

    void set_yaw(rclcpp::Parameter param)
    {
        yaw = deg2rad(param.as_double());
        mat.setEulerYPR(yaw, 0.0, 0.0);
    }

    void set_initial_position(rclcpp::Parameter param)
    {
        std::vector<double> pos = param.as_double_array();

        if (pos.size() != 3)
        {
            RCLCPP_WARN(this->get_logger(), "Position must be array of size 3, was %ld", pos.size());
            return;
        }

        set_position.setX(pos.at(0));
        set_position.setY(pos.at(1));
        set_position.setZ(pos.at(2));
        offset_position = {};
    }

    void set_initial_orientation(rclcpp::Parameter param)
    {
        std::vector<double> angles = param.as_double_array();
        if (angles.size() != 3)
        {
            RCLCPP_WARN(this->get_logger(), "Orientation must be array of size 3, was %ld", angles.size());
            return;
        }
        set_orientation.setEuler(
            deg2rad(angles.at(0)),
            deg2rad(angles.at(1)),
            deg2rad(angles.at(2))
        );

        // We need to calculate the offset new, so set this to empty
        offset_orientation = {};
    }

    tf2::Vector3 rotate_position(geometry_msgs::msg::Point position)
    {
        tf2::Vector3 pos(position.x, position.y, position.z);
        pos = mat * pos;
        return pos;
    }

    void transform_position(geometry_msgs::msg::Point &position)
    {
        if (!offset_position.has_value())
        {
            RCLCPP_ERROR(this->get_logger(), "Tried to transform position before offset was calculated!");
            return;
        }
        auto pos = rotate_position(position) + offset_position.value();

        position.x = pos.getX();
        position.y = pos.getY();
        position.z = pos.getZ();
    }

    void transform_orientation(geometry_msgs::msg::Quaternion &orientation)
    {
        if (!offset_orientation.has_value())
        {
            RCLCPP_ERROR(this->get_logger(), "Tried to transform orientation before offset was calculated!");
            return;
        }
        tf2::Quaternion q_orientation;
        tf2::fromMsg(orientation, q_orientation);
        q_orientation *= offset_orientation.value();
        q_orientation.normalize();
        orientation = tf2::toMsg(q_orientation);
    }

    void calc_offset_position(geometry_msgs::msg::Point position)
    {
        auto p_reading = rotate_position(position);
        offset_position = set_position - p_reading;
        RCLCPP_INFO(this->get_logger(), "New offset position: %f.2, %f,2 %f.2", offset_position->getX(), offset_position->getY(), offset_position->getZ());
    }

    void calc_offset_orientation(geometry_msgs::msg::Quaternion orientation)
    {
        tf2::Quaternion q_reading;
        tf2::fromMsg(orientation, q_reading);
        offset_orientation = set_orientation * q_reading.inverse();
        offset_orientation->normalize();
        RCLCPP_INFO(this->get_logger(), "New offset orientation: %f.2, %f.2, %f.2, %f.2", offset_orientation->getX(), offset_orientation->getY(), offset_orientation->getZ(), offset_orientation->getW());
    }

    void callback_odom(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        geometry_msgs::msg::Pose pose = msg->pose.pose;

        if (!offset_orientation.has_value())
        {
            calc_offset_orientation(pose.orientation);
        }

        if (!offset_position.has_value())
        {
            calc_offset_position(pose.position);
        }

        // Rotate twist
        geometry_msgs::msg::Twist twist = msg->twist.twist;
        tf2::Vector3 linear(twist.linear.x, twist.linear.y, twist.linear.z);
        tf2::Vector3 angular(twist.angular.x, twist.angular.y, twist.angular.z);
        linear = mat * linear;
        angular = mat * angular;
        twist.linear.x = linear.getX();
        twist.linear.y = linear.getY();
        twist.linear.z = linear.getZ();
        twist.angular.x = angular.getX();
        twist.angular.y = angular.getY();
        twist.angular.z = angular.getZ();

        // Rotate position
        geometry_msgs::msg::Point position = pose.position;

        RCLCPP_INFO(this->get_logger(), "Initial position: %f, %f, %f", position.x, position.y, position.z);
        transform_position(position);
        RCLCPP_INFO(this->get_logger(), "Output position: %f, %f, %f", position.x, position.y, position.z);

        geometry_msgs::msg::Quaternion orientation = pose.orientation;
        RCLCPP_INFO(this->get_logger(), "Initial orientation: %f, %f, %f, %f", orientation.x, orientation.y, orientation.z, orientation.w);
        transform_orientation(orientation);
        RCLCPP_INFO(this->get_logger(), "Output orientation: %f, %f, %f, %f", orientation.x, orientation.y, orientation.z, orientation.w);

        // Publish transformed odometry
        nav_msgs::msg::Odometry transformed_odom;

        transformed_odom.header = msg->header;
        transformed_odom.pose.pose.position = position;
        transformed_odom.pose.pose.orientation = orientation;
        transformed_odom.twist.twist = twist;

        pub_odom->publish(transformed_odom);

    }

    void callback_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        if (!offset_orientation.has_value() || !offset_orientation.has_value())
            return;

        geometry_msgs::msg::PoseWithCovariance pose = msg->pose;

        // Rotate position
        geometry_msgs::msg::Point position = pose.pose.position;
        transform_position(position);

        geometry_msgs::msg::Quaternion orientation = pose.pose.orientation;
        transform_orientation(orientation);

        geometry_msgs::msg::PoseWithCovarianceStamped transformed_pose;
        transformed_pose.header = msg->header;
        transformed_pose.pose.pose.position = position;
        transformed_pose.pose.pose.orientation = orientation;

        pub_pose->publish(transformed_pose);
    }

    void callback_path(const nav_msgs::msg::Path::SharedPtr msg)
    {
        nav_msgs::msg::Path path = *msg;
        if (!offset_position.has_value() || !offset_orientation.has_value())
            return;

        for (auto &pose : path.poses)
        {
            // Rotate position
            geometry_msgs::msg::Point position = pose.pose.position;
            transform_position(position);
            pose.pose.position = position;
        }

        // Publish transformed path
        pub_path->publish(path);
    }

    double yaw, pitch, roll;
    tf2::Matrix3x3 mat;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom;

    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path;

    std::string odom_in, odom_out, pose_in, pose_out, path_in, path_out;

    tf2::Quaternion set_orientation;
    std::optional<tf2::Quaternion> offset_orientation;

    tf2::Vector3 set_position;
    std::optional<tf2::Vector3> offset_position;
    /// The parameter event callback that will be called when a parameter is changed
    std::shared_ptr<rclcpp::Subscription<rcl_interfaces::msg::ParameterEvent,
                                         std::allocator<void>>>
        parameter_subscription;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;

    auto node = std::make_shared<Transformer>();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
