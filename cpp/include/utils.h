#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <yaml-cpp/yaml.h>


using namespace Eigen;


namespace utils
{
    int nchoosek(int n, int k);

    MatrixXd matrixPower(const MatrixXd&, int);
    SparseMatrix<double> matrixPower(const SparseMatrix<double>&, int);
    MatrixXd kroneckerProduct(const MatrixXd&, const MatrixXd&);
    SparseMatrix<double> kroneckerProduct(const SparseMatrix<double>&, const SparseMatrix<double>&);
    MatrixXd horzcat(const MatrixXd&, const MatrixXd&);
    SparseMatrix<double> horzcat(const SparseMatrix<double>&, const SparseMatrix<double>&);
    MatrixXd vertcat(const MatrixXd&, const MatrixXd&);
    SparseMatrix<double> vertcat(const SparseMatrix<double>&, const SparseMatrix<double>&);
    SparseMatrix<double> replicateSparseMatrix(const SparseMatrix<double>&, int, int);
    SparseMatrix<double> getSparseIdentity(int);
    MatrixXd blkDiag(const std::vector<MatrixXd>&);
    SparseMatrix<double> blkDiag(const std::vector<SparseMatrix<double>>&);
    void replaceSparseBlock(SparseMatrix<double>&, const MatrixXd&, int, int);
    void replaceSparseBlock(SparseMatrix<double>&, const SparseMatrix<double>&, int, int);
    std::string getExecutablePath();
    std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> loadDynamicsMatricesFromYAML(const std::string& yamlFilename);
    std::tuple<SparseMatrix<double>, SparseMatrix<double>, SparseMatrix<double>, SparseMatrix<double>> loadSparseDynamicsMatricesFromYAML(const std::string& yamlFilename);
    void addRandomPerturbation(Eigen::VectorXd& vec, double max_shift);
    VectorXd generateRandomShiftVector(int length, double max_shift);

} // end namespace utils


#endif