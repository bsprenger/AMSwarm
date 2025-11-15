#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

/**
 * @file utils.h
 * @brief Utility functions for matrix operations using Eigen library.
 *
 * Provides functions for common matrix operations such as power, Kronecker product,
 * horizontal and vertical concatenation, block diagonal, and sparse matrix replication.
 */

namespace amswarm {
namespace utils {

// Eigen type aliases
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
template<typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;

    /**
     * Calculates the binomial coefficient ("n choose k").
     * @param n The total number of items.
     * @param k The number of items to choose.
     * @return The binomial coefficient.
     */
    int nchoosek(int n, int k);

    /**
     * Computes the power of a dense matrix.
     * @param base The base matrix.
     * @param exponent The exponent to raise the base matrix to.
     * @return The resultant matrix.
     */
    MatrixXd matrixPower(const MatrixXd& base, int exponent);

    /**
     * Computes the power of a sparse matrix.
     * @param base The base sparse matrix.
     * @param exponent The exponent to raise the base matrix to.
     * @return The resultant sparse matrix.
     */
    SparseMatrix<double> matrixPower(const SparseMatrix<double>& base, int exponent);

    /**
     * Computes the Kronecker product of two dense matrices.
     * @param A The first matrix.
     * @param B The second matrix.
     * @return The Kronecker product.
     */
    MatrixXd kroneckerProduct(const MatrixXd& A, const MatrixXd& B);

    /**
     * Computes the Kronecker product of two sparse matrices.
     * @param A The first sparse matrix.
     * @param B The second sparse matrix.
     * @return The Kronecker product.
     */
    SparseMatrix<double> kroneckerProduct(const SparseMatrix<double>& A, const SparseMatrix<double>& B);

    /**
     * Concatenates two dense matrices horizontally.
     * @param A The first matrix.
     * @param B The second matrix.
     * @return The concatenated matrix.
     */
    MatrixXd horzcat(const MatrixXd& A, const MatrixXd& B);

    /**
     * Concatenates two sparse matrices horizontally.
     * @param A The first sparse matrix.
     * @param B The second sparse matrix.
     * @return The concatenated sparse matrix.
     */
    SparseMatrix<double> horzcat(const SparseMatrix<double>& A, const SparseMatrix<double>& B);

    /**
     * Concatenates two dense matrices vertically.
     * @param A The first matrix.
     * @param B The second matrix.
     * @return The concatenated matrix.
     */
    MatrixXd vertcat(const MatrixXd& A, const MatrixXd& B);

    /**
     * Concatenates two sparse matrices vertically.
     * @param A The first sparse matrix.
     * @param B The second sparse matrix.
     * @return The concatenated sparse matrix.
     */
    SparseMatrix<double> vertcat(const SparseMatrix<double>& A, const SparseMatrix<double>& B);

    /**
     * Replicates a sparse matrix to create a larger matrix by tiling it.
     * @param input The sparse matrix to replicate.
     * @param n The number of vertical replications.
     * @param m The number of horizontal replications.
     * @return The replicated sparse matrix.
     */
    SparseMatrix<double> replicateSparseMatrix(const SparseMatrix<double>& input, int n, int m);

    /**
     * Generates a sparse identity matrix of given size.
     * @param n The size of the identity matrix.
     * @return The sparse identity matrix.
     */
    SparseMatrix<double> getSparseIdentity(int n);

    /**
     * Creates a block diagonal matrix from a vector of dense matrices.
     * @param matrices A vector containing the matrices to be placed on the diagonal.
     * @return The block diagonal matrix.
     */
    MatrixXd blkDiag(const std::vector<MatrixXd>& matrices);

    /**
     * Creates a block diagonal matrix from a vector of sparse matrices.
     * @param matrices A vector containing the matrices to be placed on the diagonal.
     * @return The block diagonal sparse matrix.
     */
    SparseMatrix<double> blkDiag(const std::vector<SparseMatrix<double>>& matrices);

    /**
     * Replaces a block in a sparse matrix with another sparse matrix.
     * @param targetSparseMatrix The matrix in which the block will be replaced.
     * @param sourceSparseMatrix The matrix to insert into the target matrix.
     * @param startRow The starting row index for the block replacement.
     * @param startCol The starting column index for the block replacement.
     */
    void replaceSparseBlock(SparseMatrix<double>& targetSparseMatrix, const SparseMatrix<double>& sourceSparseMatrix, int startRow, int startCol);

} // namespace utils
} // namespace amswarm


#endif