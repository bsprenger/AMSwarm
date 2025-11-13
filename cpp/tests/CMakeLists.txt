# Test executable
add_executable(amswarm_tests
    test_basic.cpp
)

target_link_libraries(amswarm_tests
    PRIVATE
    AMSwarm
    GTest::gtest_main
)

# Discover tests
include(GoogleTest)
gtest_discover_tests(amswarm_tests)
