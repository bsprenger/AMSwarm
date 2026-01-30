# Generate protobuf sources for the library
set(PROTO_FILES
    constraint.proto
)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# AMSwarm library source files
set(AMSWARM_SOURCES
    constraint.cpp
    constraint_proto_conversion.cpp
    drone.cpp
    swarm.cpp
    utils.cpp
    ${PROTO_SRCS}
)

# Create the library target
add_library(AMSwarm SHARED ${AMSWARM_SOURCES})

# Add alias for consistent namespacing
add_library(AMSwarm::AMSwarm ALIAS AMSwarm)

# Set target properties
target_include_directories(AMSwarm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    # Include generated protobuf headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# Link dependencies
target_link_libraries(AMSwarm PUBLIC 
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    protobuf::libprotobuf
)
