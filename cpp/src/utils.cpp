#include <utils.h>
#include <iostream>
#include <unistd.h>
#include <libgen.h>  // Include the header for dirname


using namespace Eigen;


namespace utils 
{
    int nchoosek(int n, int k) {
        if (k < 0 || k > n) {
            return 0;
        }

        if (k == 0 || k == n) {
            return 1;
        }

        int result = 1;
        for (int i = 1; i <= k; i++) {
            result *= (n - i + 1);
            result /= i;
        }

        return result;
    };

    MatrixXd matrixPower(const MatrixXd& base, int exponent) {
        if (exponent == 0) {
            // Return the identity matrix for A^0
            return MatrixXd::Identity(base.rows(), base.cols());
        } else if (exponent > 0) {
            MatrixXd result = base;
            for (int i = 1; i < exponent; ++i) {
                result *= base;
            }
            return result;
        } else {
            // Handle negative exponents or other cases if needed - assume non-negative integer powers (for now)
            throw std::invalid_argument("Unsupported exponent");
        }
    }

    SparseMatrix<double> matrixPower(const SparseMatrix<double>& base, int exponent) {
        if (exponent == 0) {
            // Return the identity matrix for A^0
            SparseMatrix<double> result = SparseMatrix<double>(base.rows(), base.cols());
            result.setIdentity();
            return result;
        } else if (exponent > 0) {
            SparseMatrix<double> result = base;
            for (int i = 1; i < exponent; ++i) {
                result = result * base;
            }
            return result;
        } else {
            // Handle negative exponents or other cases if needed - assume non-negative integer powers (for now)
            throw std::invalid_argument("Unsupported exponent");
        }
    }

    // Function to perform Kronecker product for two matrices
    MatrixXd kroneckerProduct(const MatrixXd& A, const MatrixXd& B) {
        int rowsA = A.rows();
        int colsA = A.cols();
        int rowsB = B.rows();
        int colsB = B.cols();

        MatrixXd result(rowsA * rowsB, colsA * colsB);

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsA; ++j) {
                result.block(i * rowsB, j * colsB, rowsB, colsB) = A(i, j) * B;
            }
        }

        return result;
    };

    SparseMatrix<double> kroneckerProduct(const SparseMatrix<double>& A, const SparseMatrix<double>& B) {
        int rowsA = A.rows();
        int colsA = A.cols();
        int rowsB = B.rows();
        int colsB = B.cols();

        SparseMatrix<double> result(rowsA * rowsB, colsA * colsB);
        result.reserve(rowsA * rowsB * colsA * colsB);

        for (int kA = 0; kA < A.outerSize(); ++kA) {
            for (SparseMatrix<double>::InnerIterator itA(A, kA); itA; ++itA) {
                for (int kB = 0; kB < B.outerSize(); ++kB) {
                    for (SparseMatrix<double>::InnerIterator itB(B, kB); itB; ++itB) {
                        result.insert(itA.row() * rowsB + itB.row(), itA.col() * colsB + itB.col()) = itA.value() * itB.value();
                    }
                }
            }
        }

        result.makeCompressed();
        return result;
    };

    SparseMatrix<double> replicateSparseMatrix(const SparseMatrix<double>& input, int n, int m) {
        SparseMatrix<double> result(input.rows() * n, input.cols() * m);

        result.reserve(input.nonZeros() * n * m);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < input.outerSize(); ++k) {
                    for (SparseMatrix<double>::InnerIterator it(input, k); it; ++it) {
                        result.insert(i * input.rows() + it.row(), j * input.cols() + it.col()) = it.value();
                    }
                }
            }
        }

        result.makeCompressed();
        return result;
    }

    SparseMatrix<double> getSparseIdentity(int n) {
        SparseMatrix<double> result(n, n);
        result.setIdentity();
        return result;
    }

    MatrixXd blkDiag(const std::vector<MatrixXd>& matrices) {
        if (matrices.empty()) {
            // Return an empty matrix or handle the case as needed
            return MatrixXd();
        }

        // Calculate the size of the resulting block-diagonal matrix
        int rows = 0;
        int cols = 0;
        for (const auto& matrix : matrices) {
            rows += matrix.rows();
            cols += matrix.cols();
        }

        // Create a block-diagonal matrix
        MatrixXd result(rows, cols);
        int row_offset = 0;
        int col_offset = 0;
        for (const auto& matrix : matrices) {
            result.block(row_offset, col_offset, matrix.rows(), matrix.cols()) = matrix;
            row_offset += matrix.rows();
            col_offset += matrix.cols();
        }

        return result;
    }

    SparseMatrix<double> blkDiag(const std::vector<SparseMatrix<double>>& matrices) {
        if (matrices.empty()) {
            // Return an empty matrix or handle the case as needed
            return SparseMatrix<double>();
        }

        // Calculate the size of the resulting block-diagonal matrix
        int rows = 0;
        int cols = 0;
        for (const auto& matrix : matrices) {
            rows += matrix.rows();
            cols += matrix.cols();
        }

        // Create a block-diagonal matrix
        SparseMatrix<double> result(rows, cols);
        int row_offset = 0;
        int col_offset = 0;
        for (const auto& matrix : matrices) {
            for (int k = 0; k < matrix.outerSize(); ++k) {
                for (typename SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
                    result.insert(row_offset + it.row(), col_offset + it.col()) = it.value();
                }
            }
            row_offset += matrix.rows();
            col_offset += matrix.cols();
        }

        result.makeCompressed(); // Optional: compress the resulting matrix

        return result;
    }

    // Function to concatenate matrices horizontally
    MatrixXd horzcat(const MatrixXd& A, const MatrixXd& B) {
        MatrixXd result(A.rows(), A.cols() + B.cols());
        result << A, B;
        return result;
    };

    // Function to concatenate sparse matrices horizontally
    SparseMatrix<double> horzcat(const SparseMatrix<double>& A, const SparseMatrix<double>& B) {
        SparseMatrix<double> result(A.rows(), A.cols() + B.cols());
        result.reserve(A.nonZeros() + B.nonZeros());

        for (int k = 0; k < A.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                result.insert(it.row(), it.col()) = it.value();
            }
        }

        for (int k = 0; k < B.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
                result.insert(it.row(), it.col() + A.cols()) = it.value();
            }
        }

        result.makeCompressed();
        return result;
    };

    // Function to concatenate vertically
    MatrixXd vertcat(const MatrixXd& A, const MatrixXd& B) {
        MatrixXd result(A.rows() + B.rows(), A.cols());
        result << A, B;
        return result;
    };

    SparseMatrix<double> vertcat(const SparseMatrix<double>& A, const SparseMatrix<double>& B) {
        SparseMatrix<double> result(A.rows() + B.rows(), A.cols());
        result.reserve(A.nonZeros() + B.nonZeros());

        for (int k = 0; k < A.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                result.insert(it.row(), it.col()) = it.value();
            }
        }

        for (int k = 0; k < B.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
                result.insert(it.row() + A.rows(), it.col()) = it.value();
            }
        }

        result.makeCompressed();
        return result;
    }

    void replaceSparseBlock(SparseMatrix<double>& sparseMatrix, const MatrixXd& denseBlock, int startRow, int startCol) {
        // Check if the dimensions match
        assert(startRow + denseBlock.rows() <= sparseMatrix.rows());
        assert(startCol + denseBlock.cols() <= sparseMatrix.cols());

        // Iterate through the dense block
        for (int i = 0; i < denseBlock.rows(); ++i) {
            for (int j = 0; j < denseBlock.cols(); ++j) {
                // Set the value in the sparse matrix
                sparseMatrix.coeffRef(startRow + i, startCol + j) = denseBlock(i, j);
            }
        }
    }

    void replaceSparseBlock(SparseMatrix<double>& targetSparseMatrix, const SparseMatrix<double>& sourceSparseMatrix, int startRow, int startCol) {
        // Check if the dimensions match
        assert(startRow + sourceSparseMatrix.rows() <= targetSparseMatrix.rows());
        assert(startCol + sourceSparseMatrix.cols() <= targetSparseMatrix.cols());

        // Iterate through the source sparse matrix
        for (int k = 0; k < sourceSparseMatrix.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(sourceSparseMatrix, k); it; ++it) {
                // Set the value in the target sparse matrix
                targetSparseMatrix.coeffRef(startRow + it.row(), startCol + it.col()) = it.value();
            }
        }
    }


    std::string getExecutablePath() {
        char result[1024];
        ssize_t count = readlink("/proc/self/exe", result, sizeof(result));
        if (count != -1) {
            result[count] = '\0';

            // Use the dirname function to extract the folder path
            char* folderPath = dirname(result);
            return std::string(folderPath);
        } else {
            // Handle error
            return "";
        }
    }

    std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> loadDynamicsMatricesFromYAML(const std::string& yamlFilename) {
        YAML::Node config = YAML::LoadFile(yamlFilename);

        MatrixXd A, A_prime, B, B_prime;
        
        // check if dynamics is defined in yaml file
        if (config["dynamics"]) {
            YAML::Node dynamics = config["dynamics"];

            // check if A and B matrices are defined in yaml file
            if (dynamics["A"] && dynamics["B"] && dynamics["A_prime"] && dynamics["B_prime"]) {
                // get dimension of A matrix
                int num_states = dynamics["A"].size();
                
                // check if A matrix is square and A_prime is same size
                if (num_states == dynamics["A"][0].size() && num_states == dynamics["A_prime"].size() && num_states == dynamics["A_prime"][0].size()) {
                    A.resize(num_states, num_states);
                    A_prime.resize(num_states, num_states);
                    for (int i = 0; i < num_states; i++) {
                        for (int j = 0; j < num_states; j++) {
                            A(i, j) = dynamics["A"][i][j].as<double>();
                            A_prime(i, j) = dynamics["A_prime"][i][j].as<double>();
                        }
                    }
                } else {
                    throw std::runtime_error("Error: dynamics matrix A is not square or A_prime is not same size as A in " + std::string(yamlFilename));
                }

                // check if B matrix has correct number of rows, and that B_prime is the same size
                if (num_states == dynamics["B"].size() && num_states == dynamics["B_prime"].size() && dynamics["B"][0].size() == dynamics["B_prime"][0].size()) {
                    int num_inputs = dynamics["B"][0].size();
                    B.resize(num_states, num_inputs);
                    B_prime.resize(num_states, num_inputs);
                    for (int i = 0; i < num_states; i++) {
                        for (int j = 0; j < num_inputs; j++) {
                            B(i, j) = dynamics["B"][i][j].as<double>();
                            B_prime(i, j) = dynamics["B_prime"][i][j].as<double>();
                        }
                    }
                } else {
                    throw std::runtime_error("Error: dynamics matrix B has incorrect number of rows (rows should match number of states) in " + std::string(yamlFilename));
                }

            } else {
                throw std::runtime_error("Error: dynamics matrix A or B not found in " + std::string(yamlFilename));
            }
        } else {
            throw std::runtime_error("Error: dynamics not found in " + std::string(yamlFilename));
        }
        
        return std::make_tuple(A, B, A_prime, B_prime);
    };


    std::tuple<SparseMatrix<double>, SparseMatrix<double>, SparseMatrix<double>, SparseMatrix<double>> loadSparseDynamicsMatricesFromYAML(const std::string& yamlFilename) {
        YAML::Node config = YAML::LoadFile(yamlFilename);
        YAML::Node dynamics = config["dynamics"];
        int num_states = dynamics["A"].size();
        int num_inputs = dynamics["B"][0].size();
        
        // check if A matrix is square and A_prime is the same size
        SparseMatrix<double> A(num_states, num_states), A_prime(num_states, num_states);
        for (int i = 0; i < num_states; i++) {
            for (int j = 0; j < num_states; j++) {
                double value = dynamics["A"][i][j].as<double>();
                if (value != 0) {
                    A.coeffRef(i, j) = value;
                }
                value = dynamics["A_prime"][i][j].as<double>();
                if (value != 0) {
                    A_prime.coeffRef(i, j) = value;
                }
            }
        }

        SparseMatrix<double> B(num_states, num_inputs), B_prime(num_states, num_inputs);
        for (int i = 0; i < num_states; i++) {
            for (int j = 0; j < num_inputs; j++) {
                double value = dynamics["B"][i][j].as<double>();
                if (value != 0) {
                    B.coeffRef(i, j) = value;
                }
                value = dynamics["B_prime"][i][j].as<double>();
                if (value != 0) {
                    B_prime.coeffRef(i, j) = value;
                }
            }
        }
        
        return std::make_tuple(A, B, A_prime, B_prime);
    }
        
} // end namespace utils