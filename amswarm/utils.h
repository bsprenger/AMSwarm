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
    Eigen::MatrixXd matrixPower(const Eigen::MatrixXd& base, int exponent);

    /**
     * Computes the power of a sparse matrix.
     * @param base The base sparse matrix.
     * @param exponent The exponent to raise the base matrix to.
     * @return The resultant sparse matrix.
     */
    Eigen::SparseMatrix<double> matrixPower(const Eigen::SparseMatrix<double>& base, int exponent);

    /**
     * Computes the Kronecker product of two dense matrices.
     * @param A The first matrix.
     * @param B The second matrix.
     * @return The Kronecker product.
     */
    Eigen::MatrixXd kroneckerProduct(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

    /**
     * Computes the Kronecker product of two sparse matrices.
     * @param A The first sparse matrix.
     * @param B The second sparse matrix.
     * @return The Kronecker product.
     */
    Eigen::SparseMatrix<double> kroneckerProduct(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B);

    /**
     * Concatenates two dense matrices horizontally.
     * @param A The first matrix.
     * @param B The second matrix.
     * @return The concatenated matrix.
     */
    Eigen::MatrixXd horzcat(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

    /**
     * Concatenates two sparse matrices horizontally.
     * @param A The first sparse matrix.
     * @param B The second sparse matrix.
     * @return The concatenated sparse matrix.
     */
    Eigen::SparseMatrix<double> horzcat(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B);

    /**
     * Concatenates two dense matrices vertically.
     * @param A The first matrix.
     * @param B The second matrix.
     * @return The concatenated matrix.
     */
    Eigen::MatrixXd vertcat(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

    /**
     * Concatenates two sparse matrices vertically.
     * @param A The first sparse matrix.
     * @param B The second sparse matrix.
     * @return The concatenated sparse matrix.
     */
    Eigen::SparseMatrix<double> vertcat(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B);

    /**
     * Replicates a sparse matrix to create a larger matrix by tiling it.
     * @param input The sparse matrix to replicate.
     * @param n The number of vertical replications.
     * @param m The number of horizontal replications.
     * @return The replicated sparse matrix.
     */
    Eigen::SparseMatrix<double> replicateSparseMatrix(const Eigen::SparseMatrix<double>& input, int n, int m);

    /**
     * Generates a sparse identity matrix of given size.
     * @param n The size of the identity matrix.
     * @return The sparse identity matrix.
     */
    Eigen::SparseMatrix<double> getSparseIdentity(int n);

    /**
     * Creates a block diagonal matrix from a vector of dense matrices.
     * @param matrices A vector containing the matrices to be placed on the diagonal.
     * @return The block diagonal matrix.
     */
    Eigen::MatrixXd blkDiag(const std::vector<Eigen::MatrixXd>& matrices);

    /**
     * Creates a block diagonal matrix from a vector of sparse matrices.
     * @param matrices A vector containing the matrices to be placed on the diagonal.
     * @return The block diagonal sparse matrix.
     */
    Eigen::SparseMatrix<double> blkDiag(const std::vector<Eigen::SparseMatrix<double>>& matrices);

    /**
     * Replaces a block in a sparse matrix with another sparse matrix.
     * @param targetSparseMatrix The matrix in which the block will be replaced.
     * @param sourceSparseMatrix The matrix to insert into the target matrix.
     * @param startRow The starting row index for the block replacement.
     * @param startCol The starting column index for the block replacement.
     */
    void replaceSparseBlock(Eigen::SparseMatrix<double>& targetSparseMatrix, const Eigen::SparseMatrix<double>& sourceSparseMatrix, int startRow, int startCol);

} // namespace utils
} // namespace amswarm


#endif