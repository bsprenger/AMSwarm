# cmake_minimum_required(VERSION 3.0.0)
# project(AMSwarm VERSION 0.1.0 LANGUAGES C CXX)

# include(CTest)
# enable_testing()

# add_library(AMSwarm src/drone.cpp src/swarm.cpp src/simulator.cpp src/utils.cpp)


# target_include_directories(AMSwarm
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../external
# )

# # Add the -pthread flag
# set(THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads REQUIRED)
# target_link_libraries(AMSwarm PRIVATE yaml-cpp Threads::Threads)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)

cmake_minimum_required(VERSION 3.0.0)
project(AMswarm)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# cpp library files
file(GLOB AMSWARM_SOURCES "src/*.cpp")
file(GLOB AMSWARM_HEADERS "include/*.h")

add_library(AMSwarm SHARED ${AMSWARM_SOURCES} ${AMSWARM_HEADERS})
target_include_directories(AMSwarm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../external)

target_link_libraries(AMSwarm PRIVATE yaml-cpp)

# Get the current user's home directory
execute_process(COMMAND sh -c "echo $HOME" OUTPUT_VARIABLE USER_HOME OUTPUT_STRIP_TRAILING_WHITESPACE)

# Install the library, headers, and Python module
install(TARGETS AMSwarm DESTINATION "${USER_HOME}/.local/lib")
install(FILES ${AMSWARM_HEADERS} DESTINATION "${USER_HOME}/.local/include")

