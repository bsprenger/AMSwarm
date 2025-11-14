#include <gtest/gtest.h>
#include <amswarm/constraint.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

// Test EqualityConstraint construction and basic getters
TEST(EqualityConstraintTest, ConstructionAndGetters) {
    // Create a simple equality constraint: Gx = h
    SparseMatrix<double> G(2, 3);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(0, 1, 2.0));
    triplets.push_back(Triplet<double>(1, 1, 1.0));
    triplets.push_back(Triplet<double>(1, 2, 3.0));
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd h(2);
    h << 5.0, 7.0;
    
    EqualityConstraint constraint(G, h);
    
    // Test getQuadraticTerm returns const reference (G^T * G)
    const SparseMatrix<double>& Q = constraint.getQuadraticTerm();
    EXPECT_EQ(Q.rows(), 3);
    EXPECT_EQ(Q.cols(), 3);
    
    // Verify Q = G^T * G
    SparseMatrix<double> G_T = G.transpose();
    SparseMatrix<double> expected_Q = G_T * G;
    EXPECT_TRUE(Q.isApprox(expected_Q));
    
    // Test getLinearTerm returns -G^T * h
    VectorXd c = constraint.getLinearTerm();
    VectorXd expected_c = -G_T * h;
    EXPECT_TRUE(c.isApprox(expected_c));
}

// Test EqualityConstraint isSatisfied method
TEST(EqualityConstraintTest, IsSatisfied) {
    SparseMatrix<double> G(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(1, 1, 1.0));
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd h(2);
    h << 2.0, 3.0;
    
    double tolerance = 1e-2;
    EqualityConstraint constraint(G, h, tolerance);
    
    // Test with x that satisfies Gx = h
    VectorXd x_satisfied(2);
    x_satisfied << 2.0, 3.0;
    EXPECT_TRUE(constraint.isSatisfied(x_satisfied));
    
    // Test with x that nearly satisfies (within tolerance)
    VectorXd x_near(2);
    x_near << 2.005, 3.005;
    EXPECT_TRUE(constraint.isSatisfied(x_near));
    
    // Test with x that doesn't satisfy
    VectorXd x_not_satisfied(2);
    x_not_satisfied << 1.0, 1.0;
    EXPECT_FALSE(constraint.isSatisfied(x_not_satisfied));
}

// Test EqualityConstraint getBregmanUpdate method
TEST(EqualityConstraintTest, BregmanUpdate) {
    SparseMatrix<double> G(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 2.0));
    triplets.push_back(Triplet<double>(1, 1, 3.0));
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd h(2);
    h << 4.0, 6.0;
    
    EqualityConstraint constraint(G, h);
    
    VectorXd x(2);
    x << 1.0, 2.0;
    
    VectorXd update = constraint.getBregmanUpdate(x);
    
    // Verify update = G^T * G * x - G^T * h
    SparseMatrix<double> G_T = G.transpose();
    VectorXd expected_update = G_T * G * x - G_T * h;
    EXPECT_TRUE(update.isApprox(expected_update));
}

// Test InequalityConstraint construction and basic getters
TEST(InequalityConstraintTest, ConstructionAndGetters) {
    SparseMatrix<double> G(2, 3);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(0, 1, 1.0));
    triplets.push_back(Triplet<double>(1, 2, 1.0));
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd h(2);
    h << 5.0, 3.0;
    
    InequalityConstraint constraint(G, h);
    
    // Test getQuadraticTerm returns const reference (G^T * G)
    const SparseMatrix<double>& Q = constraint.getQuadraticTerm();
    EXPECT_EQ(Q.rows(), 3);
    EXPECT_EQ(Q.cols(), 3);
    
    // Verify Q = G^T * G
    SparseMatrix<double> G_T = G.transpose();
    SparseMatrix<double> expected_Q = G_T * G;
    EXPECT_TRUE(Q.isApprox(expected_Q));
}

// Test InequalityConstraint isSatisfied method
TEST(InequalityConstraintTest, IsSatisfied) {
    SparseMatrix<double> G(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(1, 1, 1.0));
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd h(2);
    h << 5.0, 5.0;
    
    double tolerance = 1e-2;
    InequalityConstraint constraint(G, h, tolerance);
    
    // Test with x that satisfies Gx <= h
    VectorXd x_satisfied(2);
    x_satisfied << 3.0, 4.0;
    EXPECT_TRUE(constraint.isSatisfied(x_satisfied));
    
    // Test with x that doesn't satisfy (Gx > h)
    VectorXd x_not_satisfied(2);
    x_not_satisfied << 6.0, 7.0;
    EXPECT_FALSE(constraint.isSatisfied(x_not_satisfied));
}

// Test InequalityConstraint update and slack variables
TEST(InequalityConstraintTest, UpdateSlack) {
    SparseMatrix<double> G(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(1, 1, 1.0));
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd h(2);
    h << 5.0, 5.0;
    
    InequalityConstraint constraint(G, h);
    
    // Before update, slack should be zero
    VectorXd c1 = constraint.getLinearTerm();
    
    // Update with x that satisfies the constraint (Gx < h, so slack becomes positive)
    VectorXd x(2);
    x << 2.0, 3.0;
    constraint.update(x);
    
    // After update with x that satisfies constraint, slack should be updated
    VectorXd c2 = constraint.getLinearTerm();
    EXPECT_FALSE(c1.isApprox(c2));
}

// Test InequalityConstraint reset method
TEST(InequalityConstraintTest, Reset) {
    SparseMatrix<double> G(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(1, 1, 1.0));
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd h(2);
    h << 5.0, 5.0;
    
    InequalityConstraint constraint(G, h);
    
    VectorXd c_initial = constraint.getLinearTerm();
    
    // Update constraint
    VectorXd x(2);
    x << 7.0, 8.0;
    constraint.update(x);
    
    // Reset constraint
    constraint.reset();
    VectorXd c_after_reset = constraint.getLinearTerm();
    
    // After reset, linear term should be same as initial
    EXPECT_TRUE(c_initial.isApprox(c_after_reset));
}

// Test PolarInequalityConstraint construction
TEST(PolarInequalityConstraintTest, Construction) {
    SparseMatrix<double> G(6, 3);  // 6 rows (2 time steps * 3 coordinates)
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < 6; ++i) {
        triplets.push_back(Triplet<double>(i, i % 3, 1.0));
    }
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd c(6);
    c.setZero();
    
    double lwr_bound = 1.0;
    double upr_bound = 10.0;
    double bf_gamma = 0.5;
    
    PolarInequalityConstraint constraint(G, c, lwr_bound, upr_bound, bf_gamma);
    
    // Test getQuadraticTerm returns const reference
    const SparseMatrix<double>& Q = constraint.getQuadraticTerm();
    EXPECT_EQ(Q.rows(), 3);
    EXPECT_EQ(Q.cols(), 3);
}

// Test PolarInequalityConstraint with invalid parameters
TEST(PolarInequalityConstraintTest, InvalidParameters) {
    SparseMatrix<double> G(6, 3);
    VectorXd c(6);
    c.setZero();
    
    // Test with invalid bf_gamma (< 0)
    EXPECT_THROW(
        PolarInequalityConstraint(G, c, 1.0, 10.0, -0.1),
        std::invalid_argument
    );
    
    // Test with invalid bf_gamma (> 1)
    EXPECT_THROW(
        PolarInequalityConstraint(G, c, 1.0, 10.0, 1.5),
        std::invalid_argument
    );
    
    // Test with invalid bounds (lwr_bound >= upr_bound)
    EXPECT_THROW(
        PolarInequalityConstraint(G, c, 10.0, 5.0, 0.5),
        std::invalid_argument
    );
}

// Test PolarInequalityConstraint update method
TEST(PolarInequalityConstraintTest, Update) {
    SparseMatrix<double> G(6, 3);
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < 6; ++i) {
        triplets.push_back(Triplet<double>(i, i % 3, 1.0));
    }
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd c(6);
    c.setZero();
    
    double lwr_bound = 1.0;
    double upr_bound = 10.0;
    
    PolarInequalityConstraint constraint(G, c, lwr_bound, upr_bound);
    
    VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    
    VectorXd c_before = constraint.getLinearTerm();
    constraint.update(x);
    VectorXd c_after = constraint.getLinearTerm();
    
    // Linear term should change after update
    EXPECT_FALSE(c_before.isApprox(c_after));
}

// Test PolarInequalityConstraint reset method
TEST(PolarInequalityConstraintTest, Reset) {
    SparseMatrix<double> G(6, 3);
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < 6; ++i) {
        triplets.push_back(Triplet<double>(i, i % 3, 1.0));
    }
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd c(6);
    c.setZero();
    
    PolarInequalityConstraint constraint(G, c, 1.0, 10.0);
    
    VectorXd c_initial = constraint.getLinearTerm();
    
    // Update constraint
    VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    constraint.update(x);
    
    // Reset constraint
    constraint.reset();
    VectorXd c_after_reset = constraint.getLinearTerm();
    
    // After reset, linear term should be same as initial
    EXPECT_TRUE(c_initial.isApprox(c_after_reset));
}

// Test that getQuadraticTerm returns a const reference (not a copy)
TEST(ConstraintTest, GetQuadraticTermReturnsReference) {
    SparseMatrix<double> G(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(1, 1, 1.0));
    G.setFromTriplets(triplets.begin(), triplets.end());
    
    VectorXd h(2);
    h << 2.0, 3.0;
    
    EqualityConstraint constraint(G, h);
    
    // Get the quadratic term multiple times
    const SparseMatrix<double>& Q1 = constraint.getQuadraticTerm();
    const SparseMatrix<double>& Q2 = constraint.getQuadraticTerm();
    
    // They should point to the same memory location (reference, not copy)
    EXPECT_EQ(&Q1, &Q2);
}
