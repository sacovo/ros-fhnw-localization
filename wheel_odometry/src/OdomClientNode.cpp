#include <iostream>
#include <cstdlib>
#include <string>
#include <cstddef>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <thread>
#include <chrono>

#include <sensor_msgs/msg/joint_state.hpp>

#include "kissnet.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/duration.hpp"

static constexpr double DEG_TO_RADIAN = 0.01745329251;
static constexpr double DECDEG_TO_RADIAN = 0.001745329251;
static constexpr double ONE_OVER_THOUSAND = 0.001;

static constexpr int WHEELS = 4;
/** preamble + wheels * (speed + position + effort) + status + error*/
static constexpr int COUNT = 1 + WHEELS * (1 + 1 + 1) + 1 + 1 + 1; // = 16
static constexpr int MESSAGE_SIZE = COUNT * sizeof(int32_t);

static constexpr int32_t MAX_TIMESTAMP =  2147483647;
static constexpr int32_t MIN_TIMESTAMP = -2147483648;

namespace kn = kissnet;

class ControllerClient
{
public:
    ControllerClient(std::string addr, rclcpp::Node::SharedPtr node, std::function<void(std::array<int32_t, COUNT>)> callback) : addr(addr), node(node), callback(callback), a_socket(addr)
    {

        a_socket.connect();
        RCLCPP_INFO(node->get_logger(), "Connected!");

        conn_thread = std::thread(&ControllerClient::open_connection, this);
        recv_thread = std::thread(&ControllerClient::recv_loop, this);
    }

    void open_connection()
    {
        while (run)
        {
            kn::socket_status status = a_socket.get_status();
            if (status != kn::socket_status::valid)
            {
                RCLCPP_ERROR(node->get_logger(), "[%s]: Socket not connected, reconnecting...", addr.c_str());
                a_socket.close();
                a_socket.connect();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    void read_values()
    {
        for (int i = 0; i < COUNT; ++i)
        {
            int value = 0;
            // Assuming little endian (change order for big endian)
            for (uint j = 0; j < sizeof(int32_t); ++j)
            {
                value |= (static_cast<uint8_t>(buffer[i * 4 + (3 - j)])) << (j * 8);
            }
            data[i] = value;
        }
    }

    void recv_loop()
    {
        while (run)
        {
            const auto start{std::chrono::steady_clock::now()};

            const auto [n, status] = a_socket.recv(buffer.data(), MESSAGE_SIZE, true);
            const auto end{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds{end - start};

            if (elapsed_seconds.count() > 0.1)
                RCLCPP_WARN(node->get_logger(), "Waited %.5f seconds", elapsed_seconds.count());

            if (n != MESSAGE_SIZE)
                RCLCPP_WARN(node->get_logger(), "Incorrect message size %ld", n);

            if (status == kn::socket_status::errored)
            {
                connected = false;
                RCLCPP_ERROR(node->get_logger(), "[%s] Error while trying to receive data", addr.c_str());
                continue;
            }
            position += n;
            if (position >= MESSAGE_SIZE)
            {
                read_values();
                callback(data);
                // Move additional data to the start of buffer
                for (int i = 0; i < position - MESSAGE_SIZE; i++)
                {
                    buffer[i] = buffer[i + MESSAGE_SIZE];
                }

                position = 0;
            }
        }
    }

    void close_connection()
    {
        run = false;
        a_socket.close();
    }

private:
    ssize_t position = 0;

    std::array<std::byte, 100> buffer;
    std::array<int32_t, COUNT> data;

    bool connected, run = true;

    std::string addr;
    rclcpp::Node::SharedPtr node;

    std::function<void(std::array<int32_t, COUNT>)> callback;

    kn::tcp_socket a_socket;
    std::thread recv_thread, conn_thread;
};

class OdomNode
{
public:
    OdomNode(rclcpp::Node::SharedPtr node) : node(node)
    {
        RCLCPP_INFO(node->get_logger(), "Starting node...");
        std::string angle_addr = "", speed_addr = "";

        angle_addr = node->get_parameter("angle_addr").as_string();
        speed_addr = node->get_parameter("speed_addr").as_string();

        if (angle_addr.length() == 0 || speed_addr.length() == 0)
        {
            throw std::runtime_error("Address not set!");
        }

        RCLCPP_INFO(node->get_logger(), "Creating publisher");

        node->declare_parameter("joint_state_topic", "/wheel/joint_states");

        publisher = node->create_publisher<sensor_msgs::msg::JointState>(
            node->get_parameter("joint_state_topic").as_string(),
            rclcpp::QoS(100));
        RCLCPP_INFO(node->get_logger(), "Creating client 1 %s", angle_addr.c_str());
        angle_client = std::make_unique<ControllerClient>(angle_addr, node, [this](std::array<int32_t, COUNT> data)
                                                          { this->callback_angle(data); });
        RCLCPP_INFO(node->get_logger(), "Creating client 2 %s", speed_addr.c_str());
        speed_client = std::make_unique<ControllerClient>(speed_addr, node, [this](std::array<int32_t, COUNT> data)
                                                          { this->callback_speed(data); });
    }

    // Units are deg, deg/s, mm/s, mA and always integers

    void publish()
    {
        msg.header.stamp = ts;
        // Build message from two arrays
        // velocity: deg/s -> radian/s
        msg.velocity = std::vector<double>{
            speed_speeds[0] * DEG_TO_RADIAN,
            speed_speeds[1] * DEG_TO_RADIAN,
            speed_speeds[2] * DEG_TO_RADIAN,
            speed_speeds[3] * DEG_TO_RADIAN,
            angle_speeds[0] * DECDEG_TO_RADIAN,
            angle_speeds[1] * DECDEG_TO_RADIAN,
            angle_speeds[2] * DECDEG_TO_RADIAN,
            angle_speeds[3] * DECDEG_TO_RADIAN,
        };

        // position: speed: deg -> radian, angle: 10 * deg -> radian
        msg.position = std::vector<double>{
            speed_positions[0] * DEG_TO_RADIAN,
            speed_positions[1] * DEG_TO_RADIAN,
            speed_positions[2] * DEG_TO_RADIAN,
            speed_positions[3] * DEG_TO_RADIAN,
            angle_positions[0] * DECDEG_TO_RADIAN,
            angle_positions[1] * DECDEG_TO_RADIAN,
            angle_positions[2] * DECDEG_TO_RADIAN,
            angle_positions[3] * DECDEG_TO_RADIAN,
        };

        // effort mA -> A
        msg.effort = std::vector<double>{
            speed_efforts[0] * ONE_OVER_THOUSAND,
            speed_efforts[1] * ONE_OVER_THOUSAND,
            speed_efforts[2] * ONE_OVER_THOUSAND,
            speed_efforts[3] * ONE_OVER_THOUSAND,
            angle_efforts[0] * ONE_OVER_THOUSAND,
            angle_efforts[1] * ONE_OVER_THOUSAND,
            angle_efforts[2] * ONE_OVER_THOUSAND,
            angle_efforts[3] * ONE_OVER_THOUSAND,
        };

        publisher->publish(msg);
        auto now = this->node->get_clock()->now();
        RCLCPP_INFO(node->get_logger(), "Published speed: %.6f [%.6f]", ts.seconds(), now.seconds());
        has_steering = false;
        has_speed = false;
    }

    void callback_speed(std::array<int32_t, COUNT> data)
    {
        // first value in data is the preamble, so + 1
        memcpy(speed_speeds, data.data() + 1, WHEELS * sizeof(int32_t));
        memcpy(speed_positions, data.data() + 1 + WHEELS, WHEELS * sizeof(int32_t));
        memcpy(speed_efforts, data.data() + 1 + 2 * WHEELS, WHEELS * sizeof(int32_t));

        // Timestamp Logic
        if (t0 == -1)
        {
            ts = this->node->get_clock()->now();
            t0 = data.at(15);
        }

        int32_t t1 = data.at(15);
        int delta;

        // Catch overflow
        if (t1 < t0)
        {
            delta = (MAX_TIMESTAMP - t0) + (t1 - MIN_TIMESTAMP);
        }
        else
        {
            delta = t1 - t0;
        }
        
        auto delta_duration = rclcpp::Duration(0, delta * 1000);

        RCLCPP_INFO(this->node->get_logger(), "T1: %.6d, %d, delta: %.6f", t1, delta, delta_duration.seconds());
        // seconds, nanseconds, but nanoseconds wrap automatically
        ts += delta_duration;
        t0 = t1;

        lock.lock();

        has_speed = true;
        if (has_steering)
        {
            publish();
        }
        lock.unlock();
    }

    void close()
    {
        angle_client->close_connection();
        speed_client->close_connection();
    }

    void callback_angle(std::array<int32_t, COUNT> data)
    {
        memcpy(angle_speeds, data.data() + 1, WHEELS * sizeof(int32_t));
        memcpy(angle_positions, data.data() + 1 + WHEELS, WHEELS * sizeof(int32_t));
        memcpy(angle_efforts, data.data() + 1 + 2 * WHEELS, WHEELS * sizeof(int32_t));

        lock.lock();
        has_steering = true;
        if (has_speed)
            publish();
        lock.unlock();
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher;

    int32_t speed_speeds[WHEELS] = {0, 0, 0, 0};
    int32_t speed_positions[WHEELS] = {0, 0, 0, 0};
    int32_t speed_efforts[WHEELS] = {0, 0, 0, 0};

    int32_t angle_speeds[WHEELS] = {0, 0, 0, 0};
    int32_t angle_positions[WHEELS] = {0, 0, 0, 0};
    int32_t angle_efforts[WHEELS] = {0, 0, 0, 0};

    sensor_msgs::msg::JointState msg;
    bool has_steering = false;
    bool has_speed = false;

    rclcpp::Time clock_init, ts;

    int32_t t0 = -1, t1;

    std::unique_ptr<ControllerClient> angle_client = nullptr, speed_client = nullptr;
    rclcpp::Node::SharedPtr node;

    std::mutex lock;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;

    auto node = std::make_shared<rclcpp::Node>("wheel_odom", options);
    node->declare_parameter("speed_addr", "");
    node->declare_parameter("angle_addr", "");

    auto client = OdomNode(node);
    rclcpp::spin(node);
    client.close();
    rclcpp::shutdown();
    return 0;
}
