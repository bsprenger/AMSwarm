#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace utils
{
    int nchoosek(int n, int k);

    Eigen::MatrixXd matrixPower(const Eigen::MatrixXd&, int);
    Eigen::SparseMatrix<double> matrixPower(const Eigen::SparseMatrix<double>&, int);
    Eigen::MatrixXd kroneckerProduct(const Eigen::MatrixXd&, const Eigen::MatrixXd&);
    Eigen::SparseMatrix<double> kroneckerProduct(const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&);
    Eigen::MatrixXd horzcat(const Eigen::MatrixXd&, const Eigen::MatrixXd&);
    Eigen::SparseMatrix<double> horzcat(const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&);
    Eigen::MatrixXd vertcat(const Eigen::MatrixXd&, const Eigen::MatrixXd&);
    Eigen::SparseMatrix<double> vertcat(const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&);
    Eigen::SparseMatrix<double> replicateSparseMatrix(const Eigen::SparseMatrix<double>&, int, int);
    Eigen::SparseMatrix<double> getSparseIdentity(int);
    Eigen::MatrixXd blkDiag(const std::vector<Eigen::MatrixXd>&);
    Eigen::SparseMatrix<double> blkDiag(const std::vector<Eigen::SparseMatrix<double>>&);
    void replaceSparseBlock(Eigen::SparseMatrix<double>&, const Eigen::MatrixXd&, int, int);
    void replaceSparseBlock(Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&, int, int);
    std::string getExecutablePath();

} // end namespace utils


#endif