cmake_minimum_required(VERSION 3.14)
project(amswarm VERSION 2.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include FetchContent module for modern dependency management
include(FetchContent)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Protobuf
find_package(Protobuf REQUIRED)

# Fetch Eigen linear algebra library
# Using Eigen 5.0.0 from the official GitLab repository
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
    GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

# Option to enable/disable tests
option(BUILD_TESTS "Build tests" OFF)
# Option to enable/disable Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
# Option to enable/disable ROS2 support
option(BUILD_ROS2_SUPPORT "Build with ROS2 support" OFF)

# ROS2 support
# ROS2 requires a full ROS2 installation and cannot be fetched via FetchContent
# due to its complex build system (ament_cmake). When BUILD_ROS2_SUPPORT is ON,
# the build expects a ROS2 installation to be sourced in the environment.
if(BUILD_ROS2_SUPPORT)
    # Find rclcpp - the ROS2 C++ client library
    find_package(rclcpp REQUIRED)
    # Find std_msgs - standard ROS2 message types
    find_package(std_msgs REQUIRED)
    
    # Export ROS2 found status for use in tests
    set(ROS2_FOUND TRUE)
    message(STATUS "ROS2 support enabled - rclcpp found")
endif()

# Set optimization level to O3 (highest level) for Release builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -fno-math-errno -DNDEBUG")
endif()

# Add subdirectories
add_subdirectory(amswarm)
add_subdirectory(bindings)

# Google Test for unit testing
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    enable_testing()
    add_subdirectory(tests)
endif()
