# Test executable
set(TEST_SOURCES
    test_utils.cpp
    test_drone.cpp
    test_swarm.cpp
    test_constraint.cpp
    test_amsolver.cpp
)

add_executable(amswarm_tests ${TEST_SOURCES})

target_link_libraries(amswarm_tests
    PRIVATE
    AMSwarm::AMSwarm
    GTest::gtest_main
)

# Discover tests
include(GoogleTest)
gtest_discover_tests(amswarm_tests)
