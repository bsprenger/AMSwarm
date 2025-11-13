cmake_minimum_required(VERSION 3.14)
project(amswarm)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include FetchContent module for dependency management
include(FetchContent)

# Try to find Eigen3 system-wide first
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  # If not found, use the bundled version in external/
  message(STATUS "Eigen3 not found system-wide, using bundled version in external/Eigen")
  add_library(Eigen3 INTERFACE)
  target_include_directories(Eigen3 INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/external")
  add_library(Eigen3::Eigen ALIAS Eigen3)
endif()

# Fetch pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found system-wide, fetching from source...")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Set optimization level to O3 (highest level)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -fno-math-errno -DNDEBUG")

# cpp library files
file(GLOB AMSWARM_SOURCES "cpp/src/*.cpp")
file(GLOB AMSWARM_HEADERS "cpp/include/*.h")

add_library(AMSwarm SHARED ${AMSWARM_SOURCES} ${AMSWARM_HEADERS})
target_include_directories(AMSwarm PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_link_libraries(AMSwarm PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

# source files for Python wrapper
file(GLOB PYTHON_WRAPPER_FILES "pybindings/*.cpp")

# Link the C++ library with pybind11
pybind11_add_module(amswarm MODULE ${PYTHON_WRAPPER_FILES})
target_include_directories(amswarm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_link_libraries(amswarm PUBLIC AMSwarm)
