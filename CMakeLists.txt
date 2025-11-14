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

# Fetch Eigen linear algebra library
# Note: Using version 3.3.7 as it's the latest available from the GitHub mirror
# Eigen 3.4.0 is only available from GitLab which may not be accessible
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
    GIT_TAG 3.3.7
    GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

# Option to enable/disable tests
option(BUILD_TESTS "Build tests" OFF)
# Option to enable/disable Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

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
