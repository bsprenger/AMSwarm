#include <gtest/gtest.h>
#include <amswarm/utils.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace utils;

// Test nchoosek function
TEST(UtilsTest, NChooseK) {
    EXPECT_EQ(nchoosek(5, 0), 1);
    EXPECT_EQ(nchoosek(5, 5), 1);
    EXPECT_EQ(nchoosek(5, 2), 10);
    EXPECT_EQ(nchoosek(10, 3), 120);
    EXPECT_EQ(nchoosek(4, 2), 6);
    
    // Edge cases
    EXPECT_EQ(nchoosek(5, -1), 0);
    EXPECT_EQ(nchoosek(5, 6), 0);
    EXPECT_EQ(nchoosek(0, 0), 1);
}

// Test dense matrix power
TEST(UtilsTest, DenseMatrixPower) {
    MatrixXd A(2, 2);
    A << 1, 2,
         3, 4;
    
    // Test A^0 = I
    MatrixXd A0 = matrixPower(A, 0);
    EXPECT_TRUE(A0.isApprox(MatrixXd::Identity(2, 2)));
    
    // Test A^1 = A
    MatrixXd A1 = matrixPower(A, 1);
    EXPECT_TRUE(A1.isApprox(A));
    
    // Test A^2
    MatrixXd A2 = matrixPower(A, 2);
    MatrixXd expected_A2(2, 2);
    expected_A2 << 7, 10,
                   15, 22;
    EXPECT_TRUE(A2.isApprox(expected_A2));
}

// Test sparse matrix power
TEST(UtilsTest, SparseMatrixPower) {
    SparseMatrix<double> A(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 2.0));
    triplets.push_back(Triplet<double>(0, 1, 1.0));
    triplets.push_back(Triplet<double>(1, 1, 3.0));
    A.setFromTriplets(triplets.begin(), triplets.end());
    
    // Test A^0 = I
    SparseMatrix<double> A0 = matrixPower(A, 0);
    SparseMatrix<double> I(2, 2);
    I.setIdentity();
    EXPECT_TRUE(A0.isApprox(I));
    
    // Test A^1 = A
    SparseMatrix<double> A1 = matrixPower(A, 1);
    EXPECT_TRUE(A1.isApprox(A));
}

// Test dense Kronecker product
TEST(UtilsTest, DenseKroneckerProduct) {
    MatrixXd A(2, 2);
    A << 1, 2,
         3, 4;
    
    MatrixXd B(2, 2);
    B << 0, 5,
         6, 7;
    
    MatrixXd result = kroneckerProduct(A, B);
    
    MatrixXd expected(4, 4);
    expected << 0, 5, 0, 10,
                6, 7, 12, 14,
                0, 15, 0, 20,
                18, 21, 24, 28;
    
    EXPECT_TRUE(result.isApprox(expected));
    EXPECT_EQ(result.rows(), A.rows() * B.rows());
    EXPECT_EQ(result.cols(), A.cols() * B.cols());
}

// Test sparse Kronecker product
TEST(UtilsTest, SparseKroneckerProduct) {
    SparseMatrix<double> A(2, 2);
    std::vector<Triplet<double>> tripletsA;
    tripletsA.push_back(Triplet<double>(0, 0, 1.0));
    tripletsA.push_back(Triplet<double>(1, 1, 2.0));
    A.setFromTriplets(tripletsA.begin(), tripletsA.end());
    
    SparseMatrix<double> B(2, 2);
    std::vector<Triplet<double>> tripletsB;
    tripletsB.push_back(Triplet<double>(0, 0, 3.0));
    tripletsB.push_back(Triplet<double>(1, 1, 4.0));
    B.setFromTriplets(tripletsB.begin(), tripletsB.end());
    
    SparseMatrix<double> result = kroneckerProduct(A, B);
    
    EXPECT_EQ(result.rows(), A.rows() * B.rows());
    EXPECT_EQ(result.cols(), A.cols() * B.cols());
    EXPECT_EQ(result.coeff(0, 0), 3.0);
    EXPECT_EQ(result.coeff(1, 1), 4.0);
    EXPECT_EQ(result.coeff(2, 2), 6.0);
    EXPECT_EQ(result.coeff(3, 3), 8.0);
}

// Test dense horizontal concatenation
TEST(UtilsTest, DenseHorzcat) {
    MatrixXd A(2, 2);
    A << 1, 2,
         3, 4;
    
    MatrixXd B(2, 3);
    B << 5, 6, 7,
         8, 9, 10;
    
    MatrixXd result = horzcat(A, B);
    
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 5);
    EXPECT_EQ(result(0, 0), 1);
    EXPECT_EQ(result(0, 2), 5);
    EXPECT_EQ(result(1, 4), 10);
}

// Test sparse horizontal concatenation
TEST(UtilsTest, SparseHorzcat) {
    SparseMatrix<double> A(2, 2);
    std::vector<Triplet<double>> tripletsA;
    tripletsA.push_back(Triplet<double>(0, 0, 1.0));
    tripletsA.push_back(Triplet<double>(1, 1, 2.0));
    A.setFromTriplets(tripletsA.begin(), tripletsA.end());
    
    SparseMatrix<double> B(2, 2);
    std::vector<Triplet<double>> tripletsB;
    tripletsB.push_back(Triplet<double>(0, 1, 3.0));
    tripletsB.push_back(Triplet<double>(1, 0, 4.0));
    B.setFromTriplets(tripletsB.begin(), tripletsB.end());
    
    SparseMatrix<double> result = horzcat(A, B);
    
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 4);
    EXPECT_EQ(result.coeff(0, 0), 1.0);
    EXPECT_EQ(result.coeff(1, 1), 2.0);
    EXPECT_EQ(result.coeff(0, 3), 3.0);
    EXPECT_EQ(result.coeff(1, 2), 4.0);
}

// Test dense vertical concatenation
TEST(UtilsTest, DenseVertcat) {
    MatrixXd A(2, 3);
    A << 1, 2, 3,
         4, 5, 6;
    
    MatrixXd B(2, 3);
    B << 7, 8, 9,
         10, 11, 12;
    
    MatrixXd result = vertcat(A, B);
    
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 3);
    EXPECT_EQ(result(0, 0), 1);
    EXPECT_EQ(result(2, 0), 7);
    EXPECT_EQ(result(3, 2), 12);
}

// Test sparse vertical concatenation
TEST(UtilsTest, SparseVertcat) {
    SparseMatrix<double> A(2, 2);
    std::vector<Triplet<double>> tripletsA;
    tripletsA.push_back(Triplet<double>(0, 0, 1.0));
    tripletsA.push_back(Triplet<double>(1, 1, 2.0));
    A.setFromTriplets(tripletsA.begin(), tripletsA.end());
    
    SparseMatrix<double> B(2, 2);
    std::vector<Triplet<double>> tripletsB;
    tripletsB.push_back(Triplet<double>(0, 1, 3.0));
    tripletsB.push_back(Triplet<double>(1, 0, 4.0));
    B.setFromTriplets(tripletsB.begin(), tripletsB.end());
    
    SparseMatrix<double> result = vertcat(A, B);
    
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 2);
    EXPECT_EQ(result.coeff(0, 0), 1.0);
    EXPECT_EQ(result.coeff(1, 1), 2.0);
    EXPECT_EQ(result.coeff(2, 1), 3.0);
    EXPECT_EQ(result.coeff(3, 0), 4.0);
}

// Test replicate sparse matrix
TEST(UtilsTest, ReplicateSparseMatrix) {
    SparseMatrix<double> input(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 1.0));
    triplets.push_back(Triplet<double>(1, 1, 2.0));
    input.setFromTriplets(triplets.begin(), triplets.end());
    
    SparseMatrix<double> result = replicateSparseMatrix(input, 2, 3);
    
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 6);
    
    // Check that pattern is replicated
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(result.coeff(i * 2 + 0, j * 2 + 0), 1.0);
            EXPECT_EQ(result.coeff(i * 2 + 1, j * 2 + 1), 2.0);
        }
    }
}

// Test sparse identity
TEST(UtilsTest, GetSparseIdentity) {
    SparseMatrix<double> I = getSparseIdentity(3);
    
    EXPECT_EQ(I.rows(), 3);
    EXPECT_EQ(I.cols(), 3);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                EXPECT_EQ(I.coeff(i, j), 1.0);
            } else {
                EXPECT_EQ(I.coeff(i, j), 0.0);
            }
        }
    }
}

// Test dense block diagonal
TEST(UtilsTest, DenseBlkDiag) {
    MatrixXd A(2, 2);
    A << 1, 2,
         3, 4;
    
    MatrixXd B(3, 3);
    B << 5, 6, 7,
         8, 9, 10,
         11, 12, 13;
    
    std::vector<MatrixXd> matrices = {A, B};
    MatrixXd result = blkDiag(matrices);
    
    EXPECT_EQ(result.rows(), 5);
    EXPECT_EQ(result.cols(), 5);
    
    // Check A block
    EXPECT_EQ(result(0, 0), 1);
    EXPECT_EQ(result(0, 1), 2);
    EXPECT_EQ(result(1, 0), 3);
    EXPECT_EQ(result(1, 1), 4);
    
    // Check B block
    EXPECT_EQ(result(2, 2), 5);
    EXPECT_EQ(result(4, 4), 13);
}

// Test sparse block diagonal
TEST(UtilsTest, SparseBlkDiag) {
    SparseMatrix<double> A(2, 2);
    std::vector<Triplet<double>> tripletsA;
    tripletsA.push_back(Triplet<double>(0, 0, 1.0));
    tripletsA.push_back(Triplet<double>(1, 1, 2.0));
    A.setFromTriplets(tripletsA.begin(), tripletsA.end());
    
    SparseMatrix<double> B(2, 2);
    std::vector<Triplet<double>> tripletsB;
    tripletsB.push_back(Triplet<double>(0, 0, 3.0));
    tripletsB.push_back(Triplet<double>(1, 1, 4.0));
    B.setFromTriplets(tripletsB.begin(), tripletsB.end());
    
    std::vector<SparseMatrix<double>> matrices = {A, B};
    SparseMatrix<double> result = blkDiag(matrices);
    
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
    EXPECT_EQ(result.coeff(0, 0), 1.0);
    EXPECT_EQ(result.coeff(1, 1), 2.0);
    EXPECT_EQ(result.coeff(2, 2), 3.0);
    EXPECT_EQ(result.coeff(3, 3), 4.0);
}

// Test replace sparse block
TEST(UtilsTest, ReplaceSparseBlock) {
    SparseMatrix<double> target(4, 4);
    target.setIdentity();
    
    SparseMatrix<double> source(2, 2);
    std::vector<Triplet<double>> triplets;
    triplets.push_back(Triplet<double>(0, 0, 5.0));
    triplets.push_back(Triplet<double>(0, 1, 6.0));
    triplets.push_back(Triplet<double>(1, 0, 7.0));
    triplets.push_back(Triplet<double>(1, 1, 8.0));
    source.setFromTriplets(triplets.begin(), triplets.end());
    
    replaceSparseBlock(target, source, 1, 1);
    
    EXPECT_EQ(target.coeff(0, 0), 1.0);
    EXPECT_EQ(target.coeff(1, 1), 5.0);
    EXPECT_EQ(target.coeff(1, 2), 6.0);
    EXPECT_EQ(target.coeff(2, 1), 7.0);
    EXPECT_EQ(target.coeff(2, 2), 8.0);
    EXPECT_EQ(target.coeff(3, 3), 1.0);
}
