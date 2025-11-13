#include <gtest/gtest.h>
#include "swarm.h"
#include "drone.h"
#include "utils.h"
#include <Eigen/Dense>

// Basic test to verify library dependencies are properly linked
TEST(BasicTest, EigenDependencyAvailable) {
    // Test that Eigen is available
    Eigen::Vector3d test_vector(1.0, 2.0, 3.0);
    EXPECT_EQ(test_vector(0), 1.0);
    EXPECT_EQ(test_vector(1), 2.0);
    EXPECT_EQ(test_vector(2), 3.0);
}

// Test that AMSwarm headers can be included
TEST(BasicTest, AMSwarmHeadersAvailable) {
    // This test simply verifies that the headers compile
    // and the library can be linked
    SUCCEED();
}

// Dummy test for initialization - can be expanded later
TEST(BasicTest, DummyTest) {
    EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
