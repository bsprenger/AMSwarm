#include <drone.h>
#include <utils.h>
#include <iostream>
#include <cmath>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>


Drone::Drone(std::string& params_filepath, Eigen::MatrixXd waypoints,
            Eigen::VectorXd initial_pos, int K, int n, float delta_t,
            Eigen::VectorXd p_min, Eigen::VectorXd p_max, float w_g_p,
            float w_g_v, float w_s, float v_bar, float f_bar)
    : waypoints(waypoints), K(K), n(n), delta_t(delta_t), p_min(p_min), 
    p_max(p_max), w_g_p(w_g_p), w_g_v(w_g_v), w_s(w_s), v_bar(v_bar),
    f_bar(f_bar), t_f(delta_t*(K-1)), W(3*K,3*(n+1)), W_dot(3*K,3*(n+1)),
    S_x(), S_u(), S_x_prime(), S_u_prime(), input_traj_vector(3 * K),
    state_traj_vector(6 * K), pos_traj_vector(3 * K), input_traj_matrix(3,K),
    state_traj_matrix(6,K), pos_traj_matrix(3,K), collision_envelope(3,3)
{   

    // initialize input parameterization (Bernstein matrices) and full horizon dynamics matrices - these will not change ever during the simulation
    generateBernsteinMatrices(); // move this out of this class
    generateFullHorizonDynamicsMatrices(params_filepath);
    // initialize collision envelope - later move this to a yaml or something
    collision_envelope.insert(0,0) = 5.8824; collision_envelope.insert(1,1) = 5.8824; collision_envelope.insert(2,2) = 2.2222;

    // initialize input trajectory to zero, we assume that no predetermined input trajectory is given
    input_traj_vector.setZero();
    input_traj_matrix.setZero();
    // initialize the state trajectory to the given initial position and zero velocity
    Eigen::VectorXd initial_state(6); initial_state << initial_pos, Eigen::VectorXd::Zero(3);
    state_traj_vector = initial_state.replicate(K,1);
    pos_traj_vector = initial_pos.replicate(K,1); // this is defined for convenience, to avoid having to extract the position from the state vector for other drones' collision avoidance
    state_traj_matrix = Eigen::Map<Eigen::MatrixXd>(state_traj_vector.data(), 6, K);
    pos_traj_matrix = Eigen::Map<Eigen::MatrixXd>(pos_traj_vector.data(), 3, K);

    initConstSelectionMatrices();
};


void Drone::solve(const double current_time, const Eigen::VectorXd x_0, const int j, std::vector<Eigen::SparseMatrix<double>> thetas, const Eigen::VectorXd xi) {

    // initialize optimization parameters
    VariableSelectionMatrices variableSelectionMatrices;
    Residuals residuals;
    LagrangeMultipliers lambda;
    Constraints constraints;
    CostMatrices costMatrices;

    Eigen::VectorXd alpha, beta, d, zeta_1, s;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    initOptimizationParams(j, x_0, xi, current_time,
                            thetas, alpha, beta, d, zeta_1, s, variableSelectionMatrices,
                            residuals, lambda, constraints, costMatrices);

    // initialize solver hyperparameters
    int max_iters = 1000;
    double rho_init = 1.3;
    double threshold = 0.01;

    // solve loop
    int iters = 0;
    while (iters < max_iters && 
            (residuals.eq.cwiseAbs().maxCoeff() > threshold ||
            residuals.pos.maxCoeff() > threshold ||
            (hard_waypoint_constraints && residuals.waypoints.cwiseAbs().maxCoeff() > threshold))) {
        
        ++iters;
        double rho = std::min(std::pow(rho_init, iters), 5.0e5);

        // STEP 1: solve for zeta_1
        computeZeta1(iters, rho, solver, costMatrices, constraints, s, lambda, zeta_1);
        

        // STEP 2: solve for alpha and beta (zeta_2)
        computeAlphaBeta(rho, constraints, zeta_1, lambda, alpha, beta, variableSelectionMatrices);
        
        // STEP 3: solve for d (zeta_3) and update h_eq with result
        compute_d(K, j, rho, constraints, zeta_1, lambda, alpha, beta, d);
        compute_h_eq(j, alpha, beta, d, constraints.h_eq);
        
        // STEP 4: update slack variable
        s = (-constraints.G_pos * zeta_1 + constraints.h_pos - lambda.pos/rho).cwiseMax(0.0);

        // STEP 5: calculate residuals and update lagrange multipliers
        computeResiduals(constraints, zeta_1, s, residuals);
        updateLagrangeMultipliers(rho, residuals, lambda);
    } // end iterative loop
    
    // calculate and return inputs and predicted trajectory
    computeInputOverHorizon(zeta_1);
    computeStatesOverHorizon(x_0);
    computePositionOverHorizon();

    // std::cout << "Waypoints in horizon:" << std::endl << extracted_waypoints << std::endl;
    // std::cout << "State traj matrix:" << std::endl << state_traj_matrix << std::endl;
    // std::cout << "Res eq: " << res_eq.cwiseAbs() << std::endl;
    // std::cout << "Res pos: " << res_pos.maxCoeff() << std::endl;
    // std::cout << "Res waypoints: " << res_waypoints.cwiseAbs().maxCoeff() << std::endl;
    // std::cout << iters << std::endl;
};


Eigen::MatrixXd Drone::extractWaypointsInCurrentHorizon(const double t, const Eigen::MatrixXd& waypoints) {
    // round all the waypoints to the nearest time step. 
    // negative time steps are allowed -- we will filter them out later
    Eigen::MatrixXd rounded_waypoints = waypoints;
    rounded_waypoints.col(0) = ((rounded_waypoints.col(0).array() - t) / delta_t).round();
    // filter out waypoints that are outside the current horizon
    std::vector<int> rows_in_horizon;
    for (int i = 0; i < rounded_waypoints.rows(); ++i) {
        if (rounded_waypoints(i, 0) >= 1 && rounded_waypoints(i, 0) <= K) {
            rows_in_horizon.push_back(i);
        }
    }

    // keep all columns
    Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(rounded_waypoints.cols(), 0,rounded_waypoints.cols());

    // return the rounded waypoints that are within the current horizon. if none, will return empty matrix. error handling elsewhere
    return rounded_waypoints(rows_in_horizon,cols);
};


void Drone::generateBernsteinMatrices() {
    float t;
    float val;
    float dot_val;

    for (int k=0;k < K;k++) { 
        t  = k*delta_t;
        float t_f_minus_t = t_f - t;
        float t_pow_n = pow(t_f,n);
        for (int m=0;m<n+1;m++) {
            val = pow(t,m)*utils::nchoosek(n,m)*pow(t_f_minus_t, n-m)/t_pow_n;

            if (k == 0 && m==0){
                dot_val = -n*pow(t_f,-1);
            } else if (k == K-1 && m == n) {
                dot_val = n*pow(t_f,-1);
            } else {
                dot_val = pow(t_f,-n)*utils::nchoosek(n,m)*(m*pow(t,m-1)*pow(t_f-t,n-m) - pow(t,m)*(n-m)*pow(t_f-t,n-m-1));
            }

            if (val == 0) { // don't bother filling in the value if zero - we are using sparse matrix
            } else {
                W.coeffRef(3*k,m) = val;
                W.coeffRef(3*k+1,m+n+1) = val;
                W.coeffRef(3*k+2,m+2*(n+1)) = val;
            }

            if (dot_val == 0) { // don't bother filling in the value if zero - we are using sparse matrix
            } else {
                W_dot.coeffRef(3*k,m) = dot_val;
                W_dot.coeffRef(3*k+1,m+n+1) = dot_val;
                W_dot.coeffRef(3*k+2,m+2*(n+1)) = dot_val;
            }
        }
    }
};


std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Drone::loadDynamicsMatricesFromFile(const std::string& yamlFilename) {
    YAML::Node config = YAML::LoadFile(yamlFilename);

    Eigen::MatrixXd A, A_prime, B, B_prime;
    
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


std::tuple<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> Drone::loadSparseDynamicsMatricesFromFile(const std::string& yamlFilename) {
    YAML::Node config = YAML::LoadFile(yamlFilename);
    YAML::Node dynamics = config["dynamics"];
    int num_states = dynamics["A"].size();
    int num_inputs = dynamics["B"][0].size();
    
    // check if A matrix is square and A_prime is the same size
    Eigen::SparseMatrix<double> A(num_states, num_states), A_prime(num_states, num_states);
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

    Eigen::SparseMatrix<double> B(num_states, num_inputs), B_prime(num_states, num_inputs);
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


void Drone::generateFullHorizonDynamicsMatrices(std::string& params_filepath) {
    // std::string executablePath = utils::getExecutablePath(); 
    std::string modelParamsYamlPath = params_filepath + "/model_params.yaml";

    // std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> matrices = loadDynamicsMatricesFromFile(modelParamsYamlPath);
    std::tuple<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> matrices = loadSparseDynamicsMatricesFromFile(modelParamsYamlPath);

    // Eigen::MatrixXd A = std::get<0>(matrices);
    // Eigen::MatrixXd B = std::get<1>(matrices);
    // Eigen::MatrixXd A_prime = std::get<2>(matrices);
    // Eigen::MatrixXd B_prime = std::get<3>(matrices);
    Eigen::SparseMatrix<double> A = std::get<0>(matrices);
    Eigen::SparseMatrix<double> B = std::get<1>(matrices);
    Eigen::SparseMatrix<double> A_prime = std::get<2>(matrices);
    Eigen::SparseMatrix<double> B_prime = std::get<3>(matrices);

    int num_states = A.rows();
    int num_inputs = B.cols();

    S_x.resize(num_states*K, num_states);
    S_x_prime.resize(num_states*K, num_states);
    S_u.resize(num_states*K, num_inputs*K);
    S_u_prime.resize(num_states*K, num_inputs*K);

    // Build S_x and S_x_prime --> to do at some point, build A_prime and B_prime from A and B, accounting for identified model time
    Eigen::SparseMatrix<double> temp_S_x_block(num_states,num_states);
    Eigen::SparseMatrix<double> temp_S_x_prime_block(num_states,num_states);
    for (int k = 0; k < K; ++k) {
        temp_S_x_block = utils::matrixPower(A,k+1); // necesssary to explicitly make this a sparse matrix to avoid ambiguous call to replaceSparseBlock
        
        temp_S_x_prime_block = A_prime * utils::matrixPower(A,k);
        utils::replaceSparseBlock(S_x, temp_S_x_block, k * num_states, 0);
        utils::replaceSparseBlock(S_x_prime, temp_S_x_prime_block, k * num_states, 0);
    }

    // Build S_u and S_u_prime
    Eigen::SparseMatrix<double> S_u_col(num_states * K, num_inputs);
    Eigen::SparseMatrix<double> S_u_prime_col(num_states * K, num_inputs);
    Eigen::SparseMatrix<double> temp_S_u_col_block(num_states,num_inputs);
    Eigen::SparseMatrix<double> temp_S_u_prime_col_block(num_states,num_inputs);
    for (int k = 0; k < K; ++k) {
        temp_S_u_col_block = utils::matrixPower(A,k) * B;
        utils::replaceSparseBlock(S_u_col, temp_S_u_col_block, k * num_states, 0);
    }

    utils::replaceSparseBlock(S_u_prime_col, B_prime, 0, 0);
    for (int k = 1; k < K; ++k) {
        temp_S_u_prime_col_block = A_prime * utils::matrixPower(A,k-1) * B;
        utils::replaceSparseBlock(S_u_prime_col, temp_S_u_prime_col_block, k * num_states, 0);
    }

    for (int k = 0; k < K; ++k) {
        utils::replaceSparseBlock(S_u, static_cast<Eigen::SparseMatrix<double>>(S_u_col.block(0, 0, (K - k) * num_states, num_inputs)), k * num_states, k * num_inputs);
        utils::replaceSparseBlock(S_u_prime, static_cast<Eigen::SparseMatrix<double>>(S_u_prime_col.block(0, 0, (K - k) * num_states, num_inputs)), k * num_states, k * num_inputs);
    }
};


void Drone::initConstSelectionMatrices() {
    // intermediate matrices used in building selection matrices
    Eigen::SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
    Eigen::SparseMatrix<double> eyeK = utils::getSparseIdentity(K);
    Eigen::SparseMatrix<double> zeroMat(3, 3);
    zeroMat.setZero();

    constSelectionMatrices.M_p = utils::kroneckerProduct(eyeK, utils::horzcat(eye3, zeroMat));
    constSelectionMatrices.M_v = utils::kroneckerProduct(eyeK, utils::horzcat(zeroMat, eye3));
    constSelectionMatrices.M_a = utils::kroneckerProduct(eyeK, utils::horzcat(zeroMat, eye3));
}

// REMOVE THIS FUNCTION FOR A CONSTRUCTOR FOR THE STRUCT
void Drone::initVariableSelectionMatrices(int j, Eigen::VectorXd& penalized_steps,
                            VariableSelectionMatrices& variableSelectionMatrices) {
    // intermediate matrices used in building selection matrices
    Eigen::SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
    Eigen::SparseMatrix<double> eye6 = utils::getSparseIdentity(6);
    Eigen::SparseMatrix<double> eyeK = utils::getSparseIdentity(K);
    Eigen::SparseMatrix<double> eyeK2j = utils::getSparseIdentity((2 + j) * K);
    Eigen::SparseMatrix<double> zeroMat(3, 3);
    zeroMat.setZero();
    Eigen::SparseMatrix<double> x_step(1,3);
    x_step.coeffRef(0,0) = 1.0; // creating a [1,0,0] row vector
    Eigen::SparseMatrix<double> y_step(1,3);
    y_step.coeffRef(0,1) = 1.0; // creating a [0,1,0] row vector
    Eigen::SparseMatrix<double> z_step(1,3);
    z_step.coeffRef(0,2) = 1.0; // creating a [0,0,1] row vector

    // create selection matrices for convenience
    variableSelectionMatrices.M_x = utils::kroneckerProduct(eyeK2j, x_step); // todo: split these out and rename them
    variableSelectionMatrices.M_y = utils::kroneckerProduct(eyeK2j, y_step);
    variableSelectionMatrices.M_z = utils::kroneckerProduct(eyeK2j, z_step);
    variableSelectionMatrices.M_waypoints_penalized.resize(6 * penalized_steps.size(), 6 * K);
    for (int i = 0; i < penalized_steps.size(); ++i) {
        utils::replaceSparseBlock(variableSelectionMatrices.M_waypoints_penalized, eye6, 6 * i, 6 * (penalized_steps(i) - 1));
    }
}


void Drone::initCostMatrices(Eigen::VectorXd& penalized_steps,
                            Eigen::VectorXd x_0,
                            Eigen::SparseMatrix<double>& X_g,
                            CostMatrices& costMatrices) {
    Eigen::SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
    Eigen::SparseMatrix<double> eyeK = utils::getSparseIdentity(K);

    std::vector<Eigen::SparseMatrix<double>> tmp_cost_vec = {eye3 * w_g_p, eye3 * w_g_v}; // clarify this
    Eigen::SparseMatrix<double> R_g = utils::blkDiag(tmp_cost_vec); // clarify this

    Eigen::SparseMatrix<double> tmp_R_g_tilde(K,K); // for selecting which steps to penalize
    for (int idx : penalized_steps) {
        tmp_R_g_tilde.insert(idx - 1, idx - 1) = 1.0; // this needs to be clarified -> since the first block in R_g_tilde corresponds to x(1), we need to subtract 1 from the index. penalized_steps gives the TIME STEP number, not matrix index
    }
    Eigen::SparseMatrix<double> R_g_tilde = utils::kroneckerProduct(tmp_R_g_tilde, R_g);
    
    // initialize R_s_tilde
    Eigen::SparseMatrix<double> R_s = eye3 * w_s;
    Eigen::SparseMatrix<double> R_s_tilde = utils::kroneckerProduct(eyeK, R_s);

    // initialize cost matrices
    costMatrices.Q = 2.0 * W.transpose() * S_u.transpose() * R_g_tilde * S_u * W  +  2.0 * W.transpose() * S_u_prime.transpose() * constSelectionMatrices.M_a.transpose() * R_s_tilde * constSelectionMatrices.M_a * S_u_prime * W;
    costMatrices.q = 2.0 * W.transpose() * S_u.transpose() * R_g_tilde.transpose() * (S_x * x_0 - X_g) + 2 * W.transpose() * S_u_prime.transpose() * constSelectionMatrices.M_a.transpose() * R_s_tilde * constSelectionMatrices.M_a * S_x_prime * x_0;
}


void Drone::computeZeta1(int iters, double rho,
                        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
                        CostMatrices& costMatrices,
                        Constraints& constraints,
                        Eigen::VectorXd& s,
                        LagrangeMultipliers& lambda,
                        Eigen::VectorXd& zeta_1) {

    costMatrices.A_check = costMatrices.Q + rho * costMatrices.A_check_const_terms;
    costMatrices.b_check = -costMatrices.q - constraints.G_eq.transpose() * lambda.eq - constraints.G_pos.transpose() * lambda.pos + rho * constraints.G_eq.transpose() * (constraints.h_eq - constraints.c_eq) + rho * constraints.G_pos.transpose() * (constraints.h_pos - s);
    if (hard_waypoint_constraints) {
        costMatrices.b_check = costMatrices.b_check - constraints.G_waypoints.transpose() * lambda.waypoints + rho * constraints.G_waypoints.transpose() * (constraints.h_waypoints - constraints.c_waypoints);
    }
    
    // Solve the sparse linear system A_check * zeta_1 = b_check
    if (iters == 1) {
        solver.analyzePattern(costMatrices.A_check);
    }
    solver.factorize(costMatrices.A_check);
    zeta_1 = solver.solve(costMatrices.b_check);
}


void Drone::compute_h_eq(int j, Eigen::VectorXd& alpha, Eigen::VectorXd& beta, Eigen::VectorXd& d,Eigen::VectorXd& h_eq) {
    // initialize omega
    Eigen::MatrixXd omega_matrix = Eigen::MatrixXd::Zero(3, (2 + j) * K); // temporary matrix to hold omega values before reshaping
    omega_matrix.row(0) = (alpha.array().cos() * beta.array().sin()).transpose();
    omega_matrix.row(1) = (alpha.array().sin() * beta.array().sin()).transpose();
    omega_matrix.row(2) = (beta.array().cos()).transpose();
    Eigen::VectorXd omega = Eigen::Map<Eigen::VectorXd>(omega_matrix.data(), omega_matrix.size()); // reshaped into vector of correct dims

    // initialize h_eq
    Eigen::MatrixXd h_eq_matrix = d.replicate(1,3).transpose();
    h_eq = Eigen::Map<Eigen::VectorXd>(h_eq_matrix.data(), h_eq_matrix.size());
    h_eq.array() *= omega.array();
}


void Drone::computeAlphaBeta(double rho, Constraints& constraints,
                            Eigen::VectorXd& zeta_1,
                            LagrangeMultipliers& lambda, Eigen::VectorXd& alpha,
                            Eigen::VectorXd& beta,
                            VariableSelectionMatrices& variableSelectionMatrices) {
    Eigen::VectorXd tmp_vec1 = variableSelectionMatrices.M_y * (constraints.G_eq * zeta_1 + constraints.c_eq + lambda.eq / rho);
    Eigen::VectorXd tmp_vec2 = variableSelectionMatrices.M_x * (constraints.G_eq * zeta_1 + constraints.c_eq + lambda.eq / rho); // the ordering of y,x,z is intentional --> clear this up later
    Eigen::VectorXd tmp_vec3 = variableSelectionMatrices.M_z * (constraints.G_eq * zeta_1 + constraints.c_eq + lambda.eq / rho);

    for (int i = 0; i < tmp_vec1.size(); ++i) {
        alpha(i) = std::atan2(tmp_vec1(i), tmp_vec2(i));
        beta(i) = std::atan2(tmp_vec2(i) / std::cos(alpha(i)), tmp_vec3(i));
    }
}


void Drone::compute_d(int K, int j, double rho,
                    Constraints& constraints, Eigen::VectorXd& zeta_1,
                    LagrangeMultipliers& lambda,
                    Eigen::VectorXd& alpha, Eigen::VectorXd& beta,
                    Eigen::VectorXd& d) {
    Eigen::MatrixXd omega_matrix = Eigen::MatrixXd::Zero(3, (2 + j) * K); // temporary matrix to hold omega values before reshaping
    omega_matrix.row(0) = (alpha.array().cos() * beta.array().sin()).transpose();
    omega_matrix.row(1) = (alpha.array().sin() * beta.array().sin()).transpose();
    omega_matrix.row(2) = (beta.array().cos()).transpose();
    Eigen::VectorXd tmp_vec4 = (constraints.G_eq * zeta_1 + constraints.c_eq + lambda.eq / rho);

    for (int i = 0; i < d.size(); ++i) {
        d(i) = tmp_vec4.segment(3 * i, 3).transpose().dot(omega_matrix.block<3, 1>(0, i));

        // clip d -- improve this later
        if (i < K) {
            d(i) = std::min(d(i), v_bar);
        } else if (i >= K && i < 2 * K) {
            d(i) = std::min(d(i), f_bar);
        } else {
            d(i) = std::max(d(i), 1.0);
        }
    }
}


void Drone::computeResiduals(Constraints& constraints,
                            Eigen::VectorXd& zeta_1, Eigen::VectorXd& s,
                            Residuals& residuals) {

    residuals.eq = constraints.G_eq * zeta_1 + constraints.c_eq - constraints.h_eq;
    residuals.pos = constraints.G_pos * zeta_1 + s - constraints.h_pos;
    residuals.waypoints = constraints.G_waypoints * zeta_1 + constraints.c_waypoints - constraints.h_waypoints;

}


void Drone::initOptimizationParams(int j,
                                    Eigen::VectorXd x_0,
                                    Eigen::VectorXd xi,
                                    double current_time,
                                    std::vector<Eigen::SparseMatrix<double>>& thetas,
                                    Eigen::VectorXd& alpha,
                                    Eigen::VectorXd& beta,
                                    Eigen::VectorXd& d,
                                    Eigen::VectorXd& zeta_1,
                                    Eigen::VectorXd& s,
                                    VariableSelectionMatrices& variableSelectionMatrices,
                                    Residuals& residuals,
                                    LagrangeMultipliers& lambda,
                                    Constraints& constraints,
                                    CostMatrices& costMatrices
                                    ) {
                                        
    // select the relevant waypoints within the current horizon (if a waypoint exists at k <= 0, ignore it, if waypoint exists > K, ignore it)
    Eigen::MatrixXd extracted_waypoints = extractWaypointsInCurrentHorizon(current_time, waypoints);
    if (extracted_waypoints.size() == 0) {
        throw std::runtime_error("Error: no waypoints within current horizon. Either increase horizon length or add waypoints.");
    }

    // extract the penalized steps from the first column of extracted_waypoints
    // note that our optimization is over x(1) to x(K). penalized_steps lines up with these indexes, i.e. the first possible penalized step is 1, NOT 0
    Eigen::VectorXd penalized_steps;
    penalized_steps.resize(extracted_waypoints.rows());
    penalized_steps = extracted_waypoints.block(0,0,extracted_waypoints.rows(),1);
    Eigen::SparseMatrix<double> X_g;
    computeX_g(extracted_waypoints, penalized_steps, X_g);
    initVariableSelectionMatrices(j, penalized_steps, variableSelectionMatrices);
    Eigen::SparseMatrix<double> S_theta = utils::blkDiag(thetas);
    initOptimizationVariables(j, alpha, beta, d, zeta_1);
    initConstConstraintMatrices(j, x_0, xi, S_theta, extracted_waypoints, variableSelectionMatrices.M_waypoints_penalized, constraints);
    compute_h_eq(j, alpha, beta, d, constraints.h_eq);
    s = Eigen::VectorXd::Zero(6 * K);
    initResiduals(j, penalized_steps.size(), residuals);
    initLagrangeMultipliers(j, penalized_steps.size(), lambda);
    initCostMatrices(penalized_steps, x_0, X_g, costMatrices);
    costMatrices.A_check_const_terms = constraints.G_eq.transpose() * constraints.G_eq + constraints.G_pos.transpose() * constraints.G_pos;
    if (hard_waypoint_constraints) {
        costMatrices.A_check_const_terms += constraints.G_waypoints.transpose() * constraints.G_waypoints;
    }
}


void Drone::initOptimizationVariables(int j, Eigen::VectorXd& alpha, Eigen::VectorXd& beta, Eigen::VectorXd& d, Eigen::VectorXd& zeta_1) {
    alpha = Eigen::VectorXd::Zero((2+j) * K);
    beta = Eigen::VectorXd::Zero((2+j) * K);
    d = Eigen::VectorXd::Zero((2+j) * K);
    zeta_1 = Eigen::VectorXd::Zero(3*(n+1));
}


void Drone::initConstConstraintMatrices(int j, Eigen::VectorXd x_0,
                                Eigen::VectorXd xi,
                                Eigen::SparseMatrix<double>& S_theta,
                                Eigen::MatrixXd& extracted_waypoints,
                                Eigen::SparseMatrix<double>& M_waypoints_penalized,
                                Constraints& constraints) {
    // everything except h_eq is constant
    // calculate waypoint constraint matrices
    constraints.G_waypoints = M_waypoints_penalized * S_u * W;
    constraints.c_waypoints = M_waypoints_penalized * S_x * x_0;
    constraints.h_waypoints = extracted_waypoints.block(0,1,extracted_waypoints.rows(),extracted_waypoints.cols()-1).reshaped<Eigen::RowMajor>();

    // Calculate G_eq
    Eigen::SparseMatrix<double> G_eq_blk1 = constSelectionMatrices.M_v * S_u * W;
    Eigen::SparseMatrix<double> G_eq_blk2 = constSelectionMatrices.M_a * S_u_prime * W;
    Eigen::SparseMatrix<double> G_eq_blk3 = S_theta * utils::replicateSparseMatrix(constSelectionMatrices.M_p * S_u * W, j, 1);
    constraints.G_eq = utils::vertcat(utils::vertcat(G_eq_blk1, G_eq_blk2), G_eq_blk3);
    
    // Calculate G_pos
    Eigen::SparseMatrix<double> G_pos_blk1 = constSelectionMatrices.M_p * S_u * W;
    Eigen::SparseMatrix<double> G_pos_blk2 = -constSelectionMatrices.M_p * S_u * W;
    constraints.G_pos = utils::vertcat(G_pos_blk1, G_pos_blk2);
    
    // Calculate h_pos
    Eigen::VectorXd h_pos_blk1 = p_max.replicate(K, 1) - constSelectionMatrices.M_p * S_x * x_0;
    Eigen::VectorXd h_pos_blk2 = -p_min.replicate(K, 1) + constSelectionMatrices.M_p * S_x * x_0;
    constraints.h_pos.resize(h_pos_blk1.rows() + h_pos_blk2.rows());
    constraints.h_pos << h_pos_blk1, h_pos_blk2;

    Eigen::VectorXd c_eq_blk1 = constSelectionMatrices.M_v * S_x * x_0;
    Eigen::VectorXd c_eq_blk2 = constSelectionMatrices.M_a * S_x_prime * x_0;
    Eigen::VectorXd c_eq_blk3 = S_theta * ( (constSelectionMatrices.M_p * S_x * x_0).replicate(j,1) - xi );
    constraints.c_eq.resize(c_eq_blk1.rows() + c_eq_blk2.rows() + c_eq_blk3.rows());
    constraints.c_eq << c_eq_blk1, c_eq_blk2, c_eq_blk3;
}


void Drone::initResiduals(int j, int num_penalized_steps, Residuals& residuals) {
    residuals.eq = Eigen::VectorXd::Ones((2 + j) * 3 * K); // TODO something more intelligent then setting these to 1 -> they should be bigger than threshold
    residuals.pos = Eigen::VectorXd::Ones(6 * K);
    residuals.waypoints = Eigen::VectorXd::Ones(6 * num_penalized_steps);
}


void Drone::initLagrangeMultipliers(int j, int num_penalized_steps, LagrangeMultipliers& lambda) {
    lambda.eq = Eigen::VectorXd::Zero((2 + j) * 3 * K);
    lambda.pos = Eigen::VectorXd::Zero(6 * K);
    lambda.waypoints = Eigen::VectorXd::Zero(6 * num_penalized_steps);
}


void Drone::computeX_g(Eigen::MatrixXd& extracted_waypoints, Eigen::VectorXd& penalized_steps, Eigen::SparseMatrix<double>& X_g) {
    X_g.resize(6 * K, 1); // position and velocity for time steps 1 to K. since no sparse vector types exist, make a sparse matrix with 1 column TODO FACT CHECK THIS!!!
    for (int i = 0; i < penalized_steps.size(); ++i) {
        Eigen::MatrixXd tmp_waypoint = extracted_waypoints.block(i,1,1,extracted_waypoints.cols()-1).transpose();
        utils::replaceSparseBlock(X_g, tmp_waypoint,(penalized_steps(i) - 1) * 6, 0); // TODO explain why we subtract 1 from penalized_steps(i)
    }
}


void Drone::updateLagrangeMultipliers(double rho, Residuals& residuals,
                                    LagrangeMultipliers& lambda) {
    lambda.eq += rho * residuals.eq;
    lambda.pos += rho * residuals.pos;
    lambda.waypoints += rho * residuals.waypoints;
}


void Drone::computeInputOverHorizon(Eigen::VectorXd& zeta_1) {
    input_traj_vector = W * zeta_1;
    input_traj_matrix = Eigen::Map<Eigen::MatrixXd>(input_traj_vector.data(), 3, K);
}


void Drone::computeStatesOverHorizon(const Eigen::VectorXd x_0) {
    state_traj_vector = S_x * x_0 + S_u * input_traj_vector;
    state_traj_matrix = Eigen::Map<Eigen::MatrixXd>(state_traj_vector.data(), 6, K);
}


void Drone::computePositionOverHorizon() {
    pos_traj_vector = constSelectionMatrices.M_p * state_traj_vector;
    pos_traj_matrix = Eigen::Map<Eigen::MatrixXd>(pos_traj_vector.data(), 3, K);
}