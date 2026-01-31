# Generate protobuf sources for testing
set(PROTO_FILES
    test_message.proto
)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Test executable
set(TEST_SOURCES
    test_utils.cpp
    test_drone.cpp
    test_swarm.cpp
    test_constraint.cpp
    test_amsolver.cpp
    test_protobuf.cpp
    ${PROTO_SRCS}
)

add_executable(amswarm_tests ${TEST_SOURCES})

# Add include directory for generated protobuf headers
target_include_directories(amswarm_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(amswarm_tests
    PRIVATE
    AMSwarm::AMSwarm
    GTest::gtest_main
    protobuf::libprotobuf
)

# Discover tests
include(GoogleTest)
gtest_discover_tests(amswarm_tests)

# ROS2 dependency tests (only built when BUILD_ROS2_SUPPORT is ON)
if(BUILD_ROS2_SUPPORT)
    add_executable(ros2_dependency_tests test_ros2_dependency.cpp)
    
    target_link_libraries(ros2_dependency_tests
        PRIVATE
        GTest::gtest_main
        rclcpp::rclcpp
        ${std_msgs_TARGETS}
    )
    
    gtest_discover_tests(ros2_dependency_tests)
endif()
