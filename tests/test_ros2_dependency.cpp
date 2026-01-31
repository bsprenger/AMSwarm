#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

// Test that rclcpp (ROS2 C++ client library) is properly linked and accessible
TEST(ROS2DependencyTest, RclcppInitShutdown) {
    // Initialize rclcpp
    rclcpp::init(0, nullptr);
    EXPECT_TRUE(rclcpp::ok());
    
    // Shutdown rclcpp
    rclcpp::shutdown();
    EXPECT_FALSE(rclcpp::ok());
}

// Test that we can create a basic ROS2 node
TEST(ROS2DependencyTest, CreateNode) {
    rclcpp::init(0, nullptr);
    
    // Create a node
    auto node = rclcpp::Node::make_shared("test_node");
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->get_name(), "test_node");
    
    rclcpp::shutdown();
}

// Test that std_msgs is properly linked
TEST(ROS2DependencyTest, StdMsgsAvailable) {
    // Create a String message
    std_msgs::msg::String msg;
    msg.data = "Hello, ROS2!";
    EXPECT_EQ(msg.data, "Hello, ROS2!");
}

