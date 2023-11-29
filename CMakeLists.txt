cmake_minimum_required(VERSION 3.0.0)
project(AMswarm)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Set optimization level to O3 (highest level)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

# cpp library files
file(GLOB AMSWARM_SOURCES "cpp/src/*.cpp")
file(GLOB AMSWARM_HEADERS "cpp/include/*.h")

add_library(AMSwarm SHARED ${AMSWARM_SOURCES} ${AMSWARM_HEADERS})
target_include_directories(AMSwarm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include ${CMAKE_CURRENT_SOURCE_DIR}/external)
target_link_libraries(AMSwarm PRIVATE yaml-cpp)

# Get the current user's home directory
execute_process(COMMAND sh -c "echo $HOME" OUTPUT_VARIABLE USER_HOME OUTPUT_STRIP_TRAILING_WHITESPACE)

# Construct the path to pybind11 based on the user's home directory
set(PYBIND11_INSTALL_PATH "${USER_HOME}/.local/lib/python3.8/site-packages")

# Add the path to pybind11 CMake module
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_INSTALL_PATH})
find_package(pybind11 REQUIRED) 

# source files for Python wrapper
file(GLOB PYTHON_WRAPPER_FILES "pybindings/*.cpp")

# Link the C++ library with pybind11
pybind11_add_module(amswarm MODULE ${PYTHON_WRAPPER_FILES})
target_include_directories(amswarm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include ${CMAKE_CURRENT_SOURCE_DIR}/external)
target_link_libraries(amswarm PUBLIC AMSwarm)

# Install the library, headers, and Python module
install(TARGETS AMSwarm DESTINATION "${USER_HOME}/.local/lib")
install(FILES ${AMSWARM_HEADERS} DESTINATION "${USER_HOME}/.local/include")


install(TARGETS amswarm
 COMPONENT python
 LIBRARY DESTINATION "${PYBIND11_INSTALL_PATH}"
 )
