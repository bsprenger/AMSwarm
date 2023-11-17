cmake_minimum_required(VERSION 3.0.0)
project(AMSwarm VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_library(AMSwarm src/drone.cpp src/swarm.cpp src/simulator.cpp src/utils.cpp)


target_include_directories(AMSwarm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external
)

# Add the -pthread flag
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(AMSwarm PRIVATE yaml-cpp Threads::Threads)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
