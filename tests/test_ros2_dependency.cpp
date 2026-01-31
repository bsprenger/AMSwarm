#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

// Test fixture for ROS2 tests that manages rclcpp context lifecycle
class ROS2DependencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!rclcpp::ok()) {
            rclcpp::init(0, nullptr);
        }
    }
    
    void TearDown() override {
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
    }
};

// Test that rclcpp (ROS2 C++ client library) is properly linked and accessible
TEST_F(ROS2DependencyTest, RclcppInitShutdown) {
    EXPECT_TRUE(rclcpp::ok());
}

// Test that we can create a basic ROS2 node
TEST_F(ROS2DependencyTest, CreateNode) {
    // Create a node
    auto node = rclcpp::Node::make_shared("test_node");
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->get_name(), "test_node");
}

// Test that std_msgs is properly linked
TEST_F(ROS2DependencyTest, StdMsgsAvailable) {
    // Create a String message
    std_msgs::msg::String msg;
    msg.data = "Hello, ROS2!";
    EXPECT_EQ(msg.data, "Hello, ROS2!");
}