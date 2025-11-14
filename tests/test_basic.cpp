#include <gtest/gtest.h>
#include <amswarm/utils.h>
#include <amswarm/swarm.h>

// Basic test to verify Google Test is working
TEST(BasicTest, GoogleTestWorks) {
    EXPECT_TRUE(true);
    EXPECT_EQ(1 + 1, 2);
}

// Test basic utilities functionality
TEST(UtilsTest, BasicFunctionality) {
    // This is a placeholder test
    // Add actual tests for your utility functions here
    EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
