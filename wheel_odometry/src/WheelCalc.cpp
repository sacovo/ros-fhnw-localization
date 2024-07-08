#include "rclcpp/rclcpp.hpp"

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

class WheelCalcNode
{
public:
    WheelCalcNode(rclcpp::Node::SharedPtr node)
    {
        node->declare_parameter("r", 0.0);
        node->declare_parameter("b", 0.0);
        node->declare_parameter("t", 0.0);

        node->get_parameter("r", r);
        node->get_parameter("b", b);
        node->get_parameter("t", t);

        twist_publisher = node->create_publisher<geometry_msgs::msg::TwistStamped>(
            node->get_parameter("twist_topic").as_string(),
            rclcpp::QoS(100));

        joint_state_subscriber = node->create_subscription<sensor_msgs::msg::JointState>(
            node->get_parameter("joint_state_topic").as_string(),
            rclcpp::QoS(100),
            std::bind(&WheelCalcNode::callback_joint_state, this, std::placeholders::_1));

        px_a = -b / 2;
        py_a = t / 2;

        px_b = b / 2;
        py_b = t / 2;

        px_c = b / 2;
        py_c = t / 2;

        px_d = -b / 2;
        py_d = -t / 2;

        clock = node->get_clock();

        RCLCPP_INFO(node->get_logger(), "Wheel positions: A(%f, %f), B(%f, %f), C(%f, %f), D(%f, %f)", px_a, py_a, px_b, py_b, px_c, py_c, px_d, py_d);
    }

    void callback_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        double w_a, w_b, w_c, w_d;
        double vx_a, vx_b, vx_c, vx_d, vy_a, vy_b, vy_c, vy_d;
        double phi_a, phi_b, phi_c, phi_d;

        double vx, vy;

        vx_a = r * w_a * cos(phi_a);
        vx_b = r * w_b * cos(phi_b);
        vx_c = r * w_c * cos(phi_c);
        vx_d = r * w_d * cos(phi_d);

        vy_a = r * w_a * sin(phi_a);
        vy_b = r * w_b * sin(phi_b);
        vy_c = r * w_c * sin(phi_c);
        vy_d = r * w_d * sin(phi_d);

        vx = 0.25 * (vx_a + vx_b + vx_c + vx_d);
        vy = 0.25 * (vy_a + vy_b + vy_c + vy_d);

        double w = (((vx_a - vx) / -py_a) + ((vy_a - vy) / px_a) + ((vx_b - vx) / -py_b) + ((vy_b - vy) / px_b) + ((vx_c - vx) / -py_c) + ((vy_c - vy) / px_c) + ((vx_d - vx) / -py_d) +
                    ((vy_d - vy) / px_d)) *
                   0.125;

        auto twist = geometry_msgs::msg::TwistStamped();
        twist.header.stamp = clock->now();
        twist.twist.linear.x = vx;
        twist.twist.linear.y = vy;
        twist.twist.angular.z = w;
        twist_publisher->publish(twist);
    }

    double r, b, t;
    // Position of the wheels on the rover
    double px_a, px_b, px_c, px_d, py_a, py_b, py_c, py_d;

    std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::TwistStamped>> twist_publisher;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::JointState>> joint_state_subscriber;
    
    std::shared_ptr<rclcpp::Clock> clock;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;

    auto node = std::make_shared<rclcpp::Node>("wheel_calc", options);

    auto calc = WheelCalcNode(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
