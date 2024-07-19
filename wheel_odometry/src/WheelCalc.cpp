#include "rclcpp/rclcpp.hpp"

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

class WheelCalcNode
{
public:
    WheelCalcNode(rclcpp::Node::SharedPtr node): node(node)
    {
        node->declare_parameter("r", 0.2);
        node->declare_parameter("b", 0.8);
        node->declare_parameter("t", 0.8);

        node->get_parameter("r", r);
        node->get_parameter("b", b);
        node->get_parameter("t", t);

        node->declare_parameter("twist_topic", "/wheel/twist");
        node->declare_parameter("joint_state_topic", "/wheel/joint_states");

        twist_publisher = node->create_publisher<geometry_msgs::msg::TwistStamped>(
            node->get_parameter("twist_topic").as_string(),
            rclcpp::QoS(100));

        RCLCPP_INFO(node->get_logger(), "Publishing twist to %s", node->get_parameter("twist_topic").as_string().c_str());

        joint_state_subscriber = node->create_subscription<sensor_msgs::msg::JointState>(
            node->get_parameter("joint_state_topic").as_string(),
            rclcpp::QoS(100),
            std::bind(&WheelCalcNode::callback_joint_state, this, std::placeholders::_1));

        RCLCPP_INFO(node->get_logger(), "Subscribing to joint states at %s", node->get_parameter("joint_state_topic").as_string().c_str());

        px_a = -b / 2;
        py_a = t / 2;

        px_b = b / 2;
        py_b = t / 2;

        px_c = b / 2;
        py_c = -t / 2;

        px_d = -b / 2;
        py_d = -t / 2;

        clock = node->get_clock();

        RCLCPP_INFO(node->get_logger(), "Wheel radius: %f, Wheel base: %f, Wheel track: %f", r, b, t); 
        RCLCPP_INFO(node->get_logger(), "Wheel positions: A(%f, %f), B(%f, %f), C(%f, %f), D(%f, %f)", px_a, py_a, px_b, py_b, px_c, py_c, px_d, py_d);
    }
    
    geometry_msgs::msg::TwistStamped calculate_twist(const sensor_msgs::msg::JointState::SharedPtr msg) {
        double w_a, w_b, w_c, w_d;
        double vx_a, vx_b, vx_c, vx_d, vy_a, vy_b, vy_c, vy_d;
        double phi_a, phi_b, phi_c, phi_d;

        w_a = msg->velocity.at(2);
        w_b = msg->velocity.at(0);
        w_c = msg->velocity.at(1);
        w_d = msg->velocity.at(3);
        
        phi_a =  msg->position.at(2 + 4);
        phi_b =  msg->position.at(0 + 4);
        phi_c =  msg->position.at(1 + 4);
        phi_d =  msg->position.at(3 + 4);

        RCLCPP_INFO(node->get_logger(), "wa: %f, wb: %f, wc: %f, wd: %f", w_a, w_b, w_c, w_d);
        RCLCPP_INFO(node->get_logger(), "pa: %f, pb: %f, pc: %f, pd: %f", phi_a, phi_b, phi_c, phi_d);

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

        RCLCPP_INFO(node->get_logger(), "vx a: %f, vy: %f", vx_a, vy_a);
        RCLCPP_INFO(node->get_logger(), "vx b: %f, vy: %f", vx_b, vy_b);
        RCLCPP_INFO(node->get_logger(), "vx c: %f, vy: %f", vx_c, vy_c);
        RCLCPP_INFO(node->get_logger(), "vx d: %f, vy: %f", vx_d, vy_d);

        RCLCPP_INFO(node->get_logger(), "vx: %f, vy: %f", vx, vy);

        double w = (
            ((vx_a - vx) / -py_a) + 
            ((vy_a - vy) / -px_a) + 
            ((vx_b - vx) / py_b) + 
            ((vy_b - vy) / px_b) + 
            ((vx_c - vx) / -py_c) + 
            ((vy_c - vy) / -px_c) + 
            ((vx_d - vx) / py_d) +
            ((vy_d - vy) / px_d)
            ) * 0.125;

        RCLCPP_INFO(node->get_logger(), "vx: %f, vy: %f, w: %f", vx, vy, w);
        auto twist = geometry_msgs::msg::TwistStamped();
        twist.header.stamp = clock->now();
        twist.header.frame_id = "wheel";
        twist.twist.linear.x = vy;
        twist.twist.linear.y = -vx;
        twist.twist.angular.z = w;

        return twist;
    }

    void callback_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        auto twist = calculate_twist(msg);
        twist_publisher->publish(twist);
    }

    double r, b, t;
    // Position of the wheels on the rover
    double px_a, px_b, px_c, px_d, py_a, py_b, py_c, py_d;

    std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::TwistStamped>> twist_publisher;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::JointState>> joint_state_subscriber;
    
    std::shared_ptr<rclcpp::Clock> clock;
    std::shared_ptr<rclcpp::Node> node;
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
