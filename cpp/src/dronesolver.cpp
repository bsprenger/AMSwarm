#include <dronesolver.h>
#include <utils.h>

#include <iostream>


DroneSolver::DroneSolver(const MPCConfig& config) {
    std::tie(W, W_dot, W_ddot, W_input) = initBernsteinMatrices(config);

    // Construct cost of the form 0.5 * x^T * quadCost * x + x^T * linearCost
    quadCost = SparseMatrix<double>(1, 1); quadCost.coeffRef(0, 0) = 2;
    linearCost = VectorXd(1); linearCost[0] = 1;

    // Construct equality constraint
    SparseMatrix<double> qc(1, 1); qc.coeffRef(0, 0) = 1;
    VectorXd lc(1); lc[0] = 2;
    std::unique_ptr<Constraint> eqConstraint = std::make_unique<EqualityConstraint>(qc, lc);
    addConstraint(std::move(eqConstraint), true);
}

void DroneSolver::preSolve(const DroneArgs& args) {
    std::cout << "Pre-solve" << std::endl;
}

DroneResult DroneSolver::postSolve(const VectorXd& x, const DroneArgs& args) {
    DroneResult result;
    result.x = x;
    return result;
}

std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> DroneSolver::initBernsteinMatrices(const MPCConfig& config) {    
    SparseMatrix<double> W(3*config.K,3*(config.n+1));
    SparseMatrix<double> W_dot(3*config.K,3*(config.n+1));
    SparseMatrix<double> W_ddot(3*config.K,3*(config.n+1));
    SparseMatrix<double> W_input(6*config.K,3*(config.n+1));

    double t_f = config.delta_t*(config.K-1);
    float t;
    float val;
    float dot_val;
    float dotdot_val;

    for (int k=0;k < config.K;k++) { 
        t  = k*config.delta_t;
        float t_f_minus_t = t_f - t;
        float t_pow_n = pow(t_f,config.n);
        for (int m=0;m<config.n+1;m++) {
            val = pow(t,m)*utils::nchoosek(config.n,m)*pow(t_f_minus_t, config.n-m)/t_pow_n;

            if (k == 0 && m == 0){
                dot_val = -config.n*pow(t_f,-1);
            } else if (k == config.K-1 && m == config.n) {
                dot_val = config.n*pow(t_f,-1);
            } else {
                dot_val = pow(t_f,-config.n)*utils::nchoosek(config.n,m)*(m*pow(t,m-1)*pow(t_f-t,config.n-m) - pow(t,m)*(config.n-m)*pow(t_f-t,config.n-m-1));
            }

            if (k == 0 && m == 0) {
                dotdot_val = config.n*(config.n-1)*pow(t_f,-2);
            } else if (k == config.K-1 && m == config.n) {
                dotdot_val = config.n*(config.n-1)*pow(t_f,-2);
            } else if (k == 0 && m == 1) {
                dotdot_val = -2*config.n*(config.n-1)*pow(t_f,-2);
            } else if (k == config.K-1 && m == config.n-1) {
                dotdot_val = -2*config.n*(config.n-1)*pow(t_f,-2);
            } else {
                dotdot_val = pow(t_f,-config.n)*utils::nchoosek(config.n,m)*(
                    m*(m-1)*pow(t,m-2)*pow(t_f-t,config.n-m)
                    -2*m*(config.n-m)*pow(t,m-1)*pow(t_f-t,config.n-m-1)
                    +(config.n-m)*(config.n-m-1)*pow(t,m)*pow(t_f-t,config.n-m-2));
            }

            if (val != 0) { // don't bother filling in the value if zero - we are using sparse matrix
                W.coeffRef(3 * k, m) = val;
                W.coeffRef(3 * k + 1, m + (config.n + 1)) = val;
                W.coeffRef(3 * k + 2, m + 2 * (config.n + 1)) = val;
            }
            if (dot_val != 0) {
                W_dot.coeffRef(3 * k, m) = dot_val;
                W_dot.coeffRef(3 * k + 1, m + (config.n + 1)) = dot_val;
                W_dot.coeffRef(3 * k + 2, m + 2 * (config.n + 1)) = dot_val;
            }
            if (dotdot_val != 0) {
                W_ddot.coeffRef(3 * k, m) = dotdot_val;
                W_ddot.coeffRef(3 * k + 1, m + (config.n + 1)) = dotdot_val;
                W_ddot.coeffRef(3 * k + 2, m + 2 * (config.n + 1)) = dotdot_val;
            }
        }
    }
    
    // construct input matrix
    for (int block = 0; block < config.K; ++block) {
        // Define the start row for each block in W and W_dot
        int startRowW = 3 * block;
        int startRowWDot = 3 * block;

        // Define the start row in W_input for W and W_dot blocks
        int startRowWInputForW = 6 * block;
        int startRowWInputForWDot = 6 * block + 3;

        // Define the block size (3 rows)
        int blockSize = 3;

        // Create submatrices for the current blocks of W and W_dot
        SparseMatrix<double> WBlock = W.block(startRowW, 0, blockSize, W.cols());
        SparseMatrix<double> WDotBlock = W_dot.block(startRowWDot, 0, blockSize, W_dot.cols());

        // Replace blocks in W_input
        utils::replaceSparseBlock(W_input, WBlock, startRowWInputForW, 0);
        utils::replaceSparseBlock(W_input, WDotBlock, startRowWInputForWDot, 0);
    }
    
    return std::make_tuple(W, W_dot, W_ddot, W_input);
};