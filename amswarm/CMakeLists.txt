# AMSwarm library source files
set(AMSWARM_SOURCES
    constraint.cpp
    drone.cpp
    swarm.cpp
    utils.cpp
)

# Generate protobuf sources
set(PROTO_FILES
    test_message.proto
)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Create the library target
add_library(AMSwarm SHARED ${AMSWARM_SOURCES} ${PROTO_SRCS})

# Add alias for consistent namespacing
add_library(AMSwarm::AMSwarm ALIAS AMSwarm)

# Set target properties
target_include_directories(AMSwarm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(AMSwarm PUBLIC 
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    protobuf::libprotobuf
)
