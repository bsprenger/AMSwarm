cmake_minimum_required(VERSION 3.0.0)
project(AMSwarm)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Get the current user's home directory
execute_process(COMMAND sh -c "echo $HOME" OUTPUT_VARIABLE USER_HOME OUTPUT_STRIP_TRAILING_WHITESPACE)

# Construct the path to pybind11 based on the user's home directory
set(PYBIND11_INSTALL_PATH "${USER_HOME}/.local/lib/python3.8/site-packages")

# Add the path to pybind11 CMake module
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_INSTALL_PATH})
find_package(pybind11 REQUIRED)

# Find Eigen3 package
find_package(Eigen3 REQUIRED)  

# source files for C++ library
file(GLOB CPP_SOURCE_FILES "cpp/src/*.cpp")
file(GLOB CPP_HEADER_FILES "cpp/include/*.hpp")

# source files for Python wrapper
file(GLOB PYTHON_WRAPPER_FILES "python/*.cpp")

# Create the C++ library (shared library)
add_library(AMSwarm SHARED ${CPP_SOURCE_FILES} ${CPP_HEADER_FILES})

# Include directories for library
target_include_directories(AMSwarm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include ${CMAKE_CURRENT_SOURCE_DIR}/external)

# Link the C++ library with pybind11
pybind11_add_module(amswarm_py ${PYTHON_WRAPPER_FILES})

# Set the output name for the shared library
set_target_properties(amswarm_py PROPERTIES OUTPUT_NAME "amswarm_py")

# Link the C++ library to the Python module
target_link_libraries(amswarm_py PRIVATE AMSwarm)

install(TARGETS amswarm_py
  COMPONENT python
  LIBRARY DESTINATION "${PYBIND11_INSTALL_PATH}/amswarm_py"
  )