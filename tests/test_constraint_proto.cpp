#include <gtest/gtest.h>
#include <amswarm/constraint.h>
#include <amswarm/constraint_proto_conversion.h>
#include "constraint.pb.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace amswarm;
using namespace amswarm::proto;
using Eigen::Triplet;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

// Helper function to create a simple sparse matrix for testing
SparseMatrix<double> createTestSparseMatrix() {
    SparseMatrix<double> G(3, 4);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.5));
    triplets.push_back(Triplet<double>(0, 2, 2.0));
    triplets.push_back(Triplet<double>(1, 1, 3.5));
    triplets.push_back(Triplet<double>(1, 3, 4.0));
    triplets.push_back(Triplet<double>(2, 0, 5.5));
    triplets.push_back(Triplet<double>(2, 2, 6.0));
    G.setFromTriplets(triplets.begin(), triplets.end());
    return G;
}

// Test round-trip conversion for EqualityConstraint
TEST(ConstraintProtoTest, EqualityConstraintRoundTrip) {
    // Create original constraint
    SparseMatrix<double> G(2, 3);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(0, 1, 2.0));
    triplets.push_back(Triplet<double>(1, 1, 1.0));
    triplets.push_back(Triplet<double>(1, 2, 3.0));
    G.setFromTriplets(triplets.begin(), triplets.end());

    VectorXd h(2);
    h << 5.0, 7.0;
    
    double tolerance = 1e-3;
    EqualityConstraint original(G, h, tolerance);

    // Convert to protobuf
    EqualityConstraintProto proto;
    toProto(original, &proto);

    // Convert back from protobuf
    EqualityConstraint restored = fromProto(proto);

    // Verify that the restored constraint has the same properties
    EXPECT_TRUE(original.getG().isApprox(restored.getG()));
    EXPECT_TRUE(original.getH().isApprox(restored.getH()));
    EXPECT_DOUBLE_EQ(original.getTolerance(), restored.getTolerance());

    // Verify that the quadratic and linear terms are the same
    EXPECT_TRUE(original.getQuadraticTerm().isApprox(restored.getQuadraticTerm()));
    EXPECT_TRUE(original.getLinearTerm().isApprox(restored.getLinearTerm()));

    // Test with a sample point
    VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    EXPECT_EQ(original.isSatisfied(x), restored.isSatisfied(x));
    EXPECT_TRUE(original.getBregmanUpdate(x).isApprox(restored.getBregmanUpdate(x)));
}

// Test round-trip conversion for InequalityConstraint
TEST(ConstraintProtoTest, InequalityConstraintRoundTrip) {
    // Create original constraint
    SparseMatrix<double> G(2, 3);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(0, 1, 1.0));
    triplets.push_back(Triplet<double>(1, 2, 1.0));
    G.setFromTriplets(triplets.begin(), triplets.end());

    VectorXd h(2);
    h << 5.0, 3.0;
    
    double tolerance = 1e-2;
    InequalityConstraint original(G, h, tolerance);

    // Update the constraint to modify slack variables
    VectorXd x(3);
    x << 2.0, 1.0, 1.5;
    original.update(x);

    // Convert to protobuf
    InequalityConstraintProto proto;
    toProto(original, &proto);

    // Convert back from protobuf
    InequalityConstraint restored = fromProto(proto);

    // Verify that the restored constraint has the same properties
    EXPECT_TRUE(original.getG().isApprox(restored.getG()));
    EXPECT_TRUE(original.getH().isApprox(restored.getH()));
    EXPECT_TRUE(original.getSlack().isApprox(restored.getSlack()));
    EXPECT_DOUBLE_EQ(original.getTolerance(), restored.getTolerance());

    // Verify that the quadratic and linear terms are the same
    EXPECT_TRUE(original.getQuadraticTerm().isApprox(restored.getQuadraticTerm()));
    EXPECT_TRUE(original.getLinearTerm().isApprox(restored.getLinearTerm()));

    // Test with a sample point
    EXPECT_EQ(original.isSatisfied(x), restored.isSatisfied(x));
    EXPECT_TRUE(original.getBregmanUpdate(x).isApprox(restored.getBregmanUpdate(x)));
}

// Test round-trip conversion for InequalityConstraint with zero slack
TEST(ConstraintProtoTest, InequalityConstraintRoundTripZeroSlack) {
    // Create original constraint (without updating)
    SparseMatrix<double> G(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(1, 1, 1.0));
    G.setFromTriplets(triplets.begin(), triplets.end());

    VectorXd h(2);
    h << 5.0, 5.0;
    
    InequalityConstraint original(G, h);

    // Convert to protobuf without updating (slack should be zero)
    InequalityConstraintProto proto;
    toProto(original, &proto);

    // Convert back from protobuf
    InequalityConstraint restored = fromProto(proto);

    // Verify slack is zero
    EXPECT_TRUE(original.getSlack().isApprox(VectorXd::Zero(2)));
    EXPECT_TRUE(restored.getSlack().isApprox(VectorXd::Zero(2)));
}

// Test round-trip conversion for PolarInequalityConstraint
TEST(ConstraintProtoTest, PolarInequalityConstraintRoundTrip) {
    // Create original constraint
    SparseMatrix<double> G(6, 3);  // 6 rows (2 time steps * 3 coordinates)
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < 6; ++i) {
        triplets.push_back(Triplet<double>(i, i % 3, 1.0 + i * 0.1));
    }
    G.setFromTriplets(triplets.begin(), triplets.end());

    VectorXd c(6);
    c << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;

    double lwr_bound = 1.0;
    double upr_bound = 10.0;
    double bf_gamma = 0.75;
    double tolerance = 1e-3;

    PolarInequalityConstraint original(G, c, lwr_bound, upr_bound, bf_gamma, tolerance);

    // Update the constraint to modify internal h
    VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    original.update(x);

    // Convert to protobuf
    PolarInequalityConstraintProto proto;
    toProto(original, &proto);

    // Convert back from protobuf
    PolarInequalityConstraint restored = fromProto(proto);

    // Verify that the restored constraint has the same properties
    EXPECT_TRUE(original.getG().isApprox(restored.getG()));
    EXPECT_TRUE(original.getC().isApprox(restored.getC()));
    EXPECT_TRUE(original.getHInternal().isApprox(restored.getHInternal()));
    EXPECT_DOUBLE_EQ(original.getLowerBound(), restored.getLowerBound());
    EXPECT_DOUBLE_EQ(original.getUpperBound(), restored.getUpperBound());
    EXPECT_DOUBLE_EQ(original.getBFGamma(), restored.getBFGamma());
    EXPECT_DOUBLE_EQ(original.getTolerance(), restored.getTolerance());

    // Verify that the quadratic and linear terms are the same
    EXPECT_TRUE(original.getQuadraticTerm().isApprox(restored.getQuadraticTerm()));
    EXPECT_TRUE(original.getLinearTerm().isApprox(restored.getLinearTerm()));

    // Test with a sample point
    EXPECT_EQ(original.isSatisfied(x), restored.isSatisfied(x));
    EXPECT_TRUE(original.getBregmanUpdate(x).isApprox(restored.getBregmanUpdate(x)));
}

// Test round-trip conversion for PolarInequalityConstraint without update
TEST(ConstraintProtoTest, PolarInequalityConstraintRoundTripNoUpdate) {
    // Create original constraint
    SparseMatrix<double> G(6, 3);
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < 6; ++i) {
        triplets.push_back(Triplet<double>(i, i % 3, 1.0));
    }
    G.setFromTriplets(triplets.begin(), triplets.end());

    VectorXd c(6);
    c.setZero();

    PolarInequalityConstraint original(G, c, 1.0, 10.0);

    // Convert to protobuf without updating
    PolarInequalityConstraintProto proto;
    toProto(original, &proto);

    // Convert back from protobuf
    PolarInequalityConstraint restored = fromProto(proto);

    // Verify that h is correctly initialized (should be -c)
    EXPECT_TRUE(original.getHInternal().isApprox(restored.getHInternal()));
}

// Test serialization and deserialization through protobuf binary format
TEST(ConstraintProtoTest, EqualityConstraintSerialization) {
    // Create original constraint
    SparseMatrix<double> G = createTestSparseMatrix();
    VectorXd h(3);
    h << 1.0, 2.0, 3.0;
    
    EqualityConstraint original(G, h, 1e-4);

    // Convert to protobuf
    EqualityConstraintProto proto;
    toProto(original, &proto);

    // Serialize to string
    std::string serialized;
    ASSERT_TRUE(proto.SerializeToString(&serialized));
    EXPECT_FALSE(serialized.empty());

    // Deserialize from string
    EqualityConstraintProto deserialized_proto;
    ASSERT_TRUE(deserialized_proto.ParseFromString(serialized));

    // Convert back to constraint
    EqualityConstraint restored = fromProto(deserialized_proto);

    // Verify equivalence
    EXPECT_TRUE(original.getG().isApprox(restored.getG()));
    EXPECT_TRUE(original.getH().isApprox(restored.getH()));
    EXPECT_DOUBLE_EQ(original.getTolerance(), restored.getTolerance());
}

// Test serialization and deserialization for InequalityConstraint through binary format
TEST(ConstraintProtoTest, InequalityConstraintSerialization) {
    // Create and update constraint
    SparseMatrix<double> G = createTestSparseMatrix();
    VectorXd h(3);
    h << 10.0, 20.0, 30.0;
    
    InequalityConstraint original(G, h, 1e-4);
    
    VectorXd x(4);
    x << 1.0, 2.0, 3.0, 4.0;
    original.update(x);

    // Convert to protobuf
    InequalityConstraintProto proto;
    toProto(original, &proto);

    // Serialize to string
    std::string serialized;
    ASSERT_TRUE(proto.SerializeToString(&serialized));

    // Deserialize and restore
    InequalityConstraintProto deserialized_proto;
    ASSERT_TRUE(deserialized_proto.ParseFromString(serialized));
    InequalityConstraint restored = fromProto(deserialized_proto);

    // Verify equivalence including internal state (slack)
    EXPECT_TRUE(original.getSlack().isApprox(restored.getSlack()));
    EXPECT_TRUE(original.getLinearTerm().isApprox(restored.getLinearTerm()));
}

// Test serialization and deserialization for PolarInequalityConstraint through binary format
TEST(ConstraintProtoTest, PolarInequalityConstraintSerialization) {
    // Create and update constraint
    SparseMatrix<double> G(9, 5);
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < 9; ++i) {
        triplets.push_back(Triplet<double>(i, i % 5, 1.0 + i * 0.05));
    }
    G.setFromTriplets(triplets.begin(), triplets.end());

    VectorXd c(9);
    for (int i = 0; i < 9; ++i) {
        c(i) = 0.1 * (i + 1);
    }

    PolarInequalityConstraint original(G, c, 2.0, 15.0, 0.8, 1e-4);
    
    VectorXd x(5);
    x << 1.0, 2.0, 3.0, 4.0, 5.0;
    original.update(x);

    // Convert to protobuf
    PolarInequalityConstraintProto proto;
    toProto(original, &proto);

    // Serialize to string
    std::string serialized;
    ASSERT_TRUE(proto.SerializeToString(&serialized));

    // Deserialize and restore
    PolarInequalityConstraintProto deserialized_proto;
    ASSERT_TRUE(deserialized_proto.ParseFromString(serialized));
    PolarInequalityConstraint restored = fromProto(deserialized_proto);

    // Verify equivalence including internal state (h)
    EXPECT_TRUE(original.getHInternal().isApprox(restored.getHInternal()));
    EXPECT_TRUE(original.getLinearTerm().isApprox(restored.getLinearTerm()));
}

// Test that reset() works correctly after deserialization
TEST(ConstraintProtoTest, InequalityConstraintResetAfterDeserialization) {
    // Create and update constraint
    SparseMatrix<double> G(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(1, 1, 1.0));
    G.setFromTriplets(triplets.begin(), triplets.end());

    VectorXd h(2);
    h << 5.0, 5.0;
    
    InequalityConstraint original(G, h);
    
    VectorXd x(2);
    x << 2.0, 3.0;
    original.update(x);

    // Serialize and deserialize
    InequalityConstraintProto proto;
    toProto(original, &proto);
    InequalityConstraint restored = fromProto(proto);

    // Reset both
    original.reset();
    restored.reset();

    // Verify both have the same state after reset
    EXPECT_TRUE(original.getSlack().isApprox(restored.getSlack()));
    EXPECT_TRUE(original.getLinearTerm().isApprox(restored.getLinearTerm()));
}

// Test that update() works correctly after deserialization
TEST(ConstraintProtoTest, PolarInequalityConstraintUpdateAfterDeserialization) {
    // Create constraint
    SparseMatrix<double> G(6, 3);
    std::vector<Triplet<double>> triplets;
    for (int i = 0; i < 6; ++i) {
        triplets.push_back(Triplet<double>(i, i % 3, 1.0));
    }
    G.setFromTriplets(triplets.begin(), triplets.end());

    VectorXd c(6);
    c.setZero();

    PolarInequalityConstraint original(G, c, 1.0, 10.0, 0.5);

    // Serialize and deserialize
    PolarInequalityConstraintProto proto;
    toProto(original, &proto);
    PolarInequalityConstraint restored = fromProto(proto);

    // Update both with the same x
    VectorXd x(3);
    x << 2.0, 3.0, 4.0;
    original.update(x);
    restored.update(x);

    // Verify both have the same state after update
    EXPECT_TRUE(original.getHInternal().isApprox(restored.getHInternal()));
    EXPECT_TRUE(original.getLinearTerm().isApprox(restored.getLinearTerm()));
}
