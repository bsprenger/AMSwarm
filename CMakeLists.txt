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

# cpp library files
file(GLOB AMSWARM_SOURCES "cpp/src/*.cpp")
file(GLOB AMSWARM_HEADERS "cpp/include/*.h")

add_library(AMSwarm SHARED ${AMSWARM_SOURCES} ${AMSWARM_HEADERS})
target_include_directories(AMSwarm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(AMSwarm PUBLIC OpenMP::OpenMP_CXX)

# pybind11 is installed since it is required by setuptools in pyproject.toml
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED) 
    
    # source files for Python wrapper
    file(GLOB PYTHON_WRAPPER_FILES "pybindings/*.cpp")
    
    # Link the C++ library with pybind11
    pybind11_add_module(amswarm MODULE ${PYTHON_WRAPPER_FILES})
    target_link_libraries(amswarm PRIVATE AMSwarm)
endif()

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
    add_subdirectory(cpp/tests)
endif()
