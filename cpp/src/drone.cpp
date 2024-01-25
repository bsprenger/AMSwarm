#include <drone.h>

#include <iostream>
#include <cmath>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>


Drone::Drone(std::string& params_filepath,
            Eigen::MatrixXd waypoints,
            MPCConfig config,
            MPCWeights weights,
            PhysicalLimits limits,
            Eigen::VectorXd initial_pos)
    : waypoints(waypoints),
    initial_pos(initial_pos),
    config(config),
    weights(weights),
    limits(limits),
    W(3*config.K,3*(config.n+1)), // TODO modify to change automatically depending on num inputs
    W_dot(3*config.K,3*(config.n+1)),
    W_ddot(3*config.K,3*(config.n+1)),
    S_x(),
    S_u(),
    S_x_prime(),
    S_u_prime(),
    collision_envelope(3,3),
    constSelectionMatrices(config.K)
{   

    // initialize input parameterization (Bernstein matrices) and full horizon dynamics matrices - these will not change ever during the simulation
    generateBernsteinMatrices(); // move to struct and constructor
    generateFullHorizonDynamicsMatrices(params_filepath); // move to struct and constructor
    
    // initialize collision envelope - later move this to a yaml or something
    collision_envelope.insert(0,0) = 5.8824; collision_envelope.insert(1,1) = 5.8824; collision_envelope.insert(2,2) = 2.2222;

};


Drone::DroneResult Drone::solve(const double current_time,
                                const Eigen::VectorXd x_0,
                                const int j,
                                std::vector<Eigen::SparseMatrix<double>> thetas,
                                const Eigen::VectorXd xi,
                                SolveOptions& opt,
                                const Eigen::VectorXd& initial_guess) {

    // extract waypoints in current horizon
    Eigen::MatrixXd extracted_waypoints = extractWaypointsInCurrentHorizon(current_time, waypoints);
    if (extracted_waypoints.size() == 0) {
        throw std::runtime_error("Error: no waypoints within current horizon. Either increase horizon length or add waypoints.");
    }

    // extract the penalized steps from the first column of extracted_waypoints
    // note that our optimization is over x(1) to x(K). penalized_steps lines up with these indexes, i.e. the first possible penalized step is 1, NOT 0
    Eigen::VectorXd penalized_steps;
    penalized_steps.resize(extracted_waypoints.rows());
    penalized_steps = extracted_waypoints.block(0,0,extracted_waypoints.rows(),1);

    // initialize optimization parameters
    VariableSelectionMatrices variableSelectionMatrices(config.K, j, penalized_steps);
    Residuals residuals(j, config.K, penalized_steps.size());
    LagrangeMultipliers lambda(j, config.K ,penalized_steps.size());
    Constraints constraints(this, j, x_0, xi, thetas, extracted_waypoints, variableSelectionMatrices);;

    // initialize optimization variables
    Eigen::VectorXd alpha, beta, d, zeta_1, s;
    alpha = Eigen::VectorXd::Zero((2+j) * config.K);
    beta = Eigen::VectorXd::Zero((2+j) * config.K);
    d = Eigen::VectorXd::Zero((2+j) * config.K);

    if (initial_guess.size() > 0) {
        zeta_1 = U_to_zeta_1(initial_guess);
    } else {
        zeta_1 = Eigen::VectorXd::Zero(3*(config.n+1));
    }
    // get the first three rows of W from Eigen and multiply by zeta_1
    Eigen::VectorXd u_0_prev = W.block(0,0,3,3*(config.n+1)) * zeta_1;
    Eigen::VectorXd u_dot_0_prev = W_dot.block(0,0,3,3*(config.n+1)) * zeta_1;
    Eigen::VectorXd u_ddot_0_prev = W_ddot.block(0,0,3,3*(config.n+1)) * zeta_1;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    
    Eigen::SparseMatrix<double> X_g;
    computeX_g(extracted_waypoints, penalized_steps, X_g);
    
    compute_h_eq(j, alpha, beta, d, constraints.h_eq);
    s = Eigen::VectorXd::Zero(6 * config.K);

    CostMatrices costMatrices(this, penalized_steps, x_0, X_g, u_0_prev,
                                u_dot_0_prev, u_ddot_0_prev, constraints,
                                opt);
    
    // solve loop
    bool solver_initialized = false;
    int iters = 0;
    while (iters < opt.max_iters && 
            (residuals.eq.cwiseAbs().maxCoeff() > opt.eq_threshold ||
            residuals.pos.maxCoeff() > opt.pos_threshold ||
            (opt.waypoint_position_constraints && residuals.waypoints_pos.cwiseAbs().maxCoeff() > opt.waypoint_position_threshold) ||
            (opt.waypoint_velocity_constraints && residuals.waypoints_vel.cwiseAbs().maxCoeff() > opt.waypoint_velocity_threshold) ||
            (opt.waypoint_acceleration_constraints && residuals.waypoints_accel.cwiseAbs().maxCoeff() > opt.waypoint_acceleration_threshold) ||
            (opt.input_continuity_constraints && residuals.input_continuity.cwiseAbs().maxCoeff() > opt.input_continuity_threshold) ||
            (opt.input_dot_continuity_constraints && residuals.input_dot_continuity.cwiseAbs().maxCoeff() > opt.input_dot_continuity_threshold) ||
            (opt.input_ddot_continuity_constraints && residuals.input_ddot_continuity.cwiseAbs().maxCoeff() > opt.input_ddot_continuity_threshold))) {
        
        ++iters;
        double rho = std::min(std::pow(opt.rho_init, iters), 5.0e5);

        // STEP 1: solve for zeta_1
        // if no initial guess is provided, then solve for zeta_1 immediately
        // if an initial guess is provided, then wait for 2nd iteration to solve for zeta_1
        if ((initial_guess.size() > 0 && iters > 1) || (initial_guess.size() == 0)) {
            updateCostMatrices(rho, costMatrices, constraints, s, lambda, opt,
                                u_0_prev, u_dot_0_prev, u_ddot_0_prev);
            if (!solver_initialized) {
                solver.analyzePattern(costMatrices.A_check);
                solver_initialized = true;
            }
            solver.factorize(costMatrices.A_check);
            zeta_1 = solver.solve(costMatrices.b_check);
        }

        // STEP 2: solve for alpha and beta (zeta_2)
        computeAlphaBeta(rho, constraints, zeta_1, lambda, alpha, beta, variableSelectionMatrices);
        
        // STEP 3: solve for d (zeta_3) and update h_eq with result
        compute_d(j, rho, constraints, zeta_1, lambda, alpha, beta, d);
        compute_h_eq(j, alpha, beta, d, constraints.h_eq);
        
        // STEP 4: update slack variable
        s = (-constraints.G_pos * zeta_1 + constraints.h_pos - lambda.pos/rho).cwiseMax(0.0);

        // STEP 5: calculate residuals and update lagrange multipliers
        computeResiduals(constraints, zeta_1, s, residuals, u_0_prev, u_dot_0_prev, u_ddot_0_prev);
        updateLagrangeMultipliers(rho, residuals, lambda);
    } // end iterative loop
    
    // calculate and return inputs and predicted trajectory
    DroneResult drone_result = computeDroneResult(current_time, zeta_1, x_0);
    if (iters < opt.max_iters) {
        drone_result.is_successful = true; // Solution found within max iterations
    } else {
        drone_result.is_successful = false; // Max iterations reached, constraints not satisfied
        printUnsatisfiedResiduals(residuals, opt);
    }
    
    return drone_result;    
};


Eigen::MatrixXd Drone::extractWaypointsInCurrentHorizon(const double t,
                                                        const Eigen::MatrixXd& waypoints) {
    // round all the waypoints to the nearest time step. 
    // negative time steps are allowed -- we will filter them out later
    Eigen::MatrixXd rounded_waypoints = waypoints;
    rounded_waypoints.col(0) = ((rounded_waypoints.col(0).array() - t) / config.delta_t).round();
    // filter out waypoints that are outside the current horizon
    std::vector<int> rows_in_horizon;
    for (int i = 0; i < rounded_waypoints.rows(); ++i) {
        if (rounded_waypoints(i, 0) >= 1 && rounded_waypoints(i, 0) <= config.K) {
            rows_in_horizon.push_back(i);
        }
    }

    // keep all columns
    Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(rounded_waypoints.cols(), 0,rounded_waypoints.cols());

    // return the rounded waypoints that are within the current horizon. if none, will return empty matrix. error handling elsewhere
    return rounded_waypoints(rows_in_horizon,cols);
};


void Drone::generateBernsteinMatrices() {
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

            if (val == 0) { // don't bother filling in the value if zero - we are using sparse matrix
            } else {
                W.coeffRef(3*k,m) = val;
                W.coeffRef(3*k+1,m+config.n+1) = val;
                W.coeffRef(3*k+2,m+2*(config.n+1)) = val;

                // W.coeffRef(6*k+3,m+3*(n+1)) = val; // TODO modify to change automatically depending on num inputs
                // W.coeffRef(6*k+4,m+4*(n+1)) = val;
                // W.coeffRef(6*k+5,m+5*(n+1)) = val;
            }

            if (dot_val == 0) { // don't bother filling in the value if zero - we are using sparse matrix
            } else {
                W_dot.coeffRef(3*k,m) = dot_val;
                W_dot.coeffRef(3*k+1,m+config.n+1) = dot_val;
                W_dot.coeffRef(3*k+2,m+2*(config.n+1)) = dot_val;

                // W_dot.coeffRef(6*k+3,m+3*(n+1)) = dot_val;
                // W_dot.coeffRef(6*k+4,m+4*(n+1)) = dot_val;
                // W_dot.coeffRef(6*k+5,m+5*(n+1)) = dot_val;
            }

            if (dotdot_val == 0) {
            } else {
                W_ddot.coeffRef(3*k,m) = dotdot_val;
                W_ddot.coeffRef(3*k+1,m+config.n+1) = dotdot_val;
                W_ddot.coeffRef(3*k+2,m+2*(config.n+1)) = dotdot_val;

                // W_ddot.coeffRef(6*k+3,m+3*(n+1)) = dotdot_val;
                // W_ddot.coeffRef(6*k+4,m+4*(n+1)) = dotdot_val;
                // W_ddot.coeffRef(6*k+5,m+5*(n+1)) = dotdot_val;
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

    S_x.resize(num_states*config.K, num_states);
    S_x_prime.resize(num_states*config.K, num_states);
    S_u.resize(num_states*config.K, num_inputs*config.K);
    S_u_prime.resize(num_states*config.K, num_inputs*config.K);

    // Build S_x and S_x_prime --> to do at some point, build A_prime and B_prime from A and B, accounting for identified model time
    Eigen::SparseMatrix<double> temp_S_x_block(num_states,num_states);
    Eigen::SparseMatrix<double> temp_S_x_prime_block(num_states,num_states);
    for (int k = 0; k < config.K; ++k) {
        temp_S_x_block = utils::matrixPower(A,k+1); // necesssary to explicitly make this a sparse matrix to avoid ambiguous call to replaceSparseBlock
        
        temp_S_x_prime_block = A_prime * utils::matrixPower(A,k);
        utils::replaceSparseBlock(S_x, temp_S_x_block, k * num_states, 0);
        utils::replaceSparseBlock(S_x_prime, temp_S_x_prime_block, k * num_states, 0);
    }

    // Build S_u and S_u_prime
    Eigen::SparseMatrix<double> S_u_col(num_states * config.K, num_inputs);
    Eigen::SparseMatrix<double> S_u_prime_col(num_states * config.K, num_inputs);
    Eigen::SparseMatrix<double> temp_S_u_col_block(num_states,num_inputs);
    Eigen::SparseMatrix<double> temp_S_u_prime_col_block(num_states,num_inputs);
    for (int k = 0; k < config.K; ++k) {
        temp_S_u_col_block = utils::matrixPower(A,k) * B;
        utils::replaceSparseBlock(S_u_col, temp_S_u_col_block, k * num_states, 0);
    }

    utils::replaceSparseBlock(S_u_prime_col, B_prime, 0, 0);
    for (int k = 1; k < config.K; ++k) {
        temp_S_u_prime_col_block = A_prime * utils::matrixPower(A,k-1) * B;
        utils::replaceSparseBlock(S_u_prime_col, temp_S_u_prime_col_block, k * num_states, 0);
    }

    for (int k = 0; k < config.K; ++k) {
        utils::replaceSparseBlock(S_u, static_cast<Eigen::SparseMatrix<double>>(S_u_col.block(0, 0, (config.K - k) * num_states, num_inputs)), k * num_states, k * num_inputs);
        utils::replaceSparseBlock(S_u_prime, static_cast<Eigen::SparseMatrix<double>>(S_u_prime_col.block(0, 0, (config.K - k) * num_states, num_inputs)), k * num_states, k * num_inputs);
    }
};



void Drone::updateCostMatrices(double rho, CostMatrices& costMatrices,
                            Constraints& constraints,
                            Eigen::VectorXd& s,
                            LagrangeMultipliers& lambda,
                            SolveOptions& opt,
                            Eigen::VectorXd& u_0_prev,
                            Eigen::VectorXd& u_dot_0_prev,
                            Eigen::VectorXd& u_ddot_0_prev) {

    costMatrices.A_check = costMatrices.Q + rho * costMatrices.A_check_const_terms;
    costMatrices.b_check = -costMatrices.q - constraints.G_eq.transpose() * lambda.eq - constraints.G_pos.transpose() * lambda.pos + rho * constraints.G_eq.transpose() * (constraints.h_eq - constraints.c_eq) + rho * constraints.G_pos.transpose() * (constraints.h_pos - s);
    if (opt.waypoint_position_constraints) {
        costMatrices.b_check = costMatrices.b_check - constraints.G_waypoints_pos.transpose() * lambda.waypoints_pos + rho * constraints.G_waypoints_pos.transpose() * (constraints.h_waypoints_pos - constraints.c_waypoints_pos);
    }
    if (opt.waypoint_velocity_constraints) {
        costMatrices.b_check = costMatrices.b_check - constraints.G_waypoints_vel.transpose() * lambda.waypoints_vel + rho * constraints.G_waypoints_vel.transpose() * (constraints.h_waypoints_vel - constraints.c_waypoints_vel);
    }
    if (opt.waypoint_acceleration_constraints) {
        costMatrices.b_check = costMatrices.b_check - constraints.G_waypoints_accel.transpose() * lambda.waypoints_accel + rho * constraints.G_waypoints_accel.transpose() * (constraints.h_waypoints_accel - constraints.c_waypoints_accel);
    }
    if (opt.input_continuity_constraints) {
        costMatrices.b_check = costMatrices.b_check - W.block(0,0,3,3*(config.n+1)).transpose() * lambda.input_continuity + rho * W.block(0,0,3,3*(config.n+1)).transpose() * u_0_prev;
    }
    if (opt.input_dot_continuity_constraints) {
        costMatrices.b_check = costMatrices.b_check - W_dot.block(0,0,3,3*(config.n+1)).transpose() * lambda.input_dot_continuity + rho * W_dot.block(0,0,3,3*(config.n+1)).transpose() * u_dot_0_prev;
    }
    if (opt.input_ddot_continuity_constraints) {
        costMatrices.b_check = costMatrices.b_check - W_ddot.block(0,0,3,3*(config.n+1)).transpose() * lambda.input_ddot_continuity + rho * W_ddot.block(0,0,3,3*(config.n+1)).transpose() * u_ddot_0_prev;
    }

}


void Drone::compute_h_eq(int j, Eigen::VectorXd& alpha, Eigen::VectorXd& beta, Eigen::VectorXd& d,Eigen::VectorXd& h_eq) {
    // initialize omega
    Eigen::MatrixXd omega_matrix = Eigen::MatrixXd::Zero(3, (2 + j) * config.K); // temporary matrix to hold omega values before reshaping
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


void Drone::compute_d(int j, double rho,
                    Constraints& constraints, Eigen::VectorXd& zeta_1,
                    LagrangeMultipliers& lambda,
                    Eigen::VectorXd& alpha, Eigen::VectorXd& beta,
                    Eigen::VectorXd& d) {
    Eigen::MatrixXd omega_matrix = Eigen::MatrixXd::Zero(3, (2 + j) * config.K); // temporary matrix to hold omega values before reshaping
    omega_matrix.row(0) = (alpha.array().cos() * beta.array().sin()).transpose();
    omega_matrix.row(1) = (alpha.array().sin() * beta.array().sin()).transpose();
    omega_matrix.row(2) = (beta.array().cos()).transpose();
    Eigen::VectorXd tmp_vec4 = (constraints.G_eq * zeta_1 + constraints.c_eq + lambda.eq / rho);

    for (int i = 0; i < d.size(); ++i) {
        d(i) = tmp_vec4.segment(3 * i, 3).transpose().dot(omega_matrix.block<3, 1>(0, i));

        // clip d -- improve this later
        if (i < config.K) {
            d(i) = std::min(d(i), limits.v_bar);
        } else if (i >= config.K && i < 2 * config.K) {
            d(i) = std::min(d(i), limits.f_bar);
        } else {
            d(i) = std::max(d(i), 1.0);
        }
    }
}


void Drone::computeResiduals(Constraints& constraints,
                            Eigen::VectorXd& zeta_1, Eigen::VectorXd& s,
                            Residuals& residuals,
                            Eigen::VectorXd& u_0_prev,
                            Eigen::VectorXd& u_dot_0_prev,
                            Eigen::VectorXd& u_ddot_0_prev) {

    residuals.eq = constraints.G_eq * zeta_1 + constraints.c_eq - constraints.h_eq;
    residuals.pos = constraints.G_pos * zeta_1 + s - constraints.h_pos;
    residuals.waypoints_pos = constraints.G_waypoints_pos * zeta_1 + constraints.c_waypoints_pos - constraints.h_waypoints_pos;
    residuals.waypoints_vel = constraints.G_waypoints_vel * zeta_1 + constraints.c_waypoints_vel - constraints.h_waypoints_vel;
    residuals.waypoints_accel = constraints.G_waypoints_accel * zeta_1 + constraints.c_waypoints_accel - constraints.h_waypoints_accel;
    residuals.input_continuity = W.block(0,0,3,3*(config.n+1)) * zeta_1 - u_0_prev;
    residuals.input_dot_continuity = W_dot.block(0,0,3,3*(config.n+1)) * zeta_1 - u_dot_0_prev;
    residuals.input_ddot_continuity = W_ddot.block(0,0,3,3*(config.n+1)) * zeta_1 - u_ddot_0_prev;
}


void Drone::computeX_g(Eigen::MatrixXd& extracted_waypoints, Eigen::VectorXd& penalized_steps, Eigen::SparseMatrix<double>& X_g) {
    X_g.resize(6 * config.K, 1); // position and velocity for time steps 1 to K. since no sparse vector types exist, make a sparse matrix with 1 column TODO FACT CHECK THIS!!!
    for (int i = 0; i < penalized_steps.size(); ++i) {
        Eigen::MatrixXd tmp_waypoint = extracted_waypoints.block(i,1,1,extracted_waypoints.cols()-1).transpose();
        utils::replaceSparseBlock(X_g, tmp_waypoint,(penalized_steps(i) - 1) * 6, 0); // TODO explain why we subtract 1 from penalized_steps(i)
    }
}


void Drone::updateLagrangeMultipliers(double rho, Residuals& residuals,
                                    LagrangeMultipliers& lambda) {
    lambda.eq += rho * residuals.eq;
    lambda.pos += rho * residuals.pos;
    lambda.waypoints_pos += rho * residuals.waypoints_pos;
    lambda.waypoints_vel += rho * residuals.waypoints_vel;
    lambda.waypoints_accel += rho * residuals.waypoints_accel;
    lambda.input_continuity += rho * residuals.input_continuity;
    lambda.input_dot_continuity += rho * residuals.input_dot_continuity;
    lambda.input_ddot_continuity += rho * residuals.input_ddot_continuity;
}


Drone::DroneResult Drone::computeDroneResult(double current_time, Eigen::VectorXd& zeta_1, Eigen::VectorXd x_0) {
    DroneResult drone_result;

    // input trajectory
    drone_result.control_input_trajectory_vector = W * zeta_1;
    drone_result.control_input_trajectory = Eigen::Map<Eigen::MatrixXd>(drone_result.control_input_trajectory_vector.data(), 3, config.K).transpose(); // TODO automatically resize based on num inputs

    // state trajectory
    drone_result.state_trajectory_vector = S_x * x_0 + S_u * drone_result.control_input_trajectory_vector;
    drone_result.state_trajectory = Eigen::Map<Eigen::MatrixXd>(drone_result.state_trajectory_vector.data(), 6, config.K).transpose();

    // position trajectory
    drone_result.position_trajectory_vector = constSelectionMatrices.M_p * drone_result.state_trajectory_vector;
    drone_result.position_trajectory = Eigen::Map<Eigen::MatrixXd>(drone_result.position_trajectory_vector.data(), 3, config.K).transpose();

    // Time stamps for position and state trajectories
    drone_result.position_state_time_stamps.resize(config.K);
    for (int i = 0; i < config.K; ++i) {
        drone_result.position_state_time_stamps(i) = current_time + (i+1) * config.delta_t;
    }

    // Time stamps for control input trajectory
    drone_result.control_input_time_stamps.resize(config.K);
    for (int i = 0; i < config.K; ++i) {
        drone_result.control_input_time_stamps(i) = current_time + i * config.delta_t;
    }

    return drone_result;
}

void Drone::printUnsatisfiedResiduals(const Residuals& residuals,
                                    SolveOptions& opt) {
    // Helper function to print steps where residuals exceed threshold
    auto printExceedingSteps = [this](const Eigen::VectorXd& residual, int start, int end, const std::string& message, double threshold, bool wrap = false) {
        std::vector<int> exceedingSteps;
        for (int i = start; i < end; i += 3) { // Iterate in steps of 3
            if (std::abs(residual[i]) > threshold || (i + 1 < end && std::abs(residual[i+1]) > threshold) || (i + 2 < end && std::abs(residual[i+2]) > threshold)) {
                int step = (i - start) / 3 + 1;
                if (wrap) {
                    step = (step - 1) % config.K + 1; // Wrap step number if it exceeds K
                }
                exceedingSteps.push_back(step); // Adjust for relative step number
            }
        }
        if (!exceedingSteps.empty()) {
            std::cout << message << " at steps: ";
            for (size_t i = 0; i < exceedingSteps.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << exceedingSteps[i];
            }
            std::cout << std::endl;
        }
    };

    // Check and print for each type of constraint
    if (residuals.eq.cwiseAbs().maxCoeff() > opt.eq_threshold) {
        printExceedingSteps(residuals.eq, 0, 3*config.K, "Velocity constraints residual exceeds threshold", opt.eq_threshold);
        printExceedingSteps(residuals.eq, 3*config.K, 6*config.K, "Acceleration constraints residual exceeds threshold", opt.eq_threshold);
        // For collision constraints, wrap the step number if it exceeds K
        printExceedingSteps(residuals.eq, 6*config.K, residuals.eq.size(), "Collision constraints residual exceeds threshold", opt.eq_threshold, true);
    }

    if (residuals.pos.maxCoeff() > opt.pos_threshold) {
        printExceedingSteps(residuals.pos, 0, residuals.pos.size(), "Position constraints residual exceeds threshold", opt.pos_threshold);
    }

    if (opt.waypoint_position_constraints && residuals.waypoints_pos.cwiseAbs().maxCoeff() > opt.waypoint_position_threshold) {
        printExceedingSteps(residuals.waypoints_pos, 0, residuals.waypoints_pos.size(), "Waypoint position constraints residual exceeds threshold", opt.waypoint_position_threshold);
    }

    if (opt.waypoint_velocity_constraints && residuals.waypoints_vel.cwiseAbs().maxCoeff() > opt.waypoint_velocity_threshold) {
        printExceedingSteps(residuals.waypoints_vel, 0, residuals.waypoints_vel.size(), "Waypoint velocity constraints residual exceeds threshold", opt.waypoint_velocity_threshold);
    }

    if (opt.waypoint_acceleration_constraints && residuals.waypoints_accel.cwiseAbs().maxCoeff() > opt.waypoint_acceleration_threshold) {
        printExceedingSteps(residuals.waypoints_accel, 0, residuals.waypoints_accel.size(), "Waypoint acceleration constraints residual exceeds threshold", opt.waypoint_acceleration_threshold);
    }
}


Eigen::VectorXd Drone::U_to_zeta_1(const Eigen::VectorXd& U) {
    Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(W);
    solver.factorize(W);
    Eigen::VectorXd zeta_1 = solver.solve(U);
    return zeta_1;
}


Eigen::VectorXd Drone::getInitialPosition() {
    return initial_pos;
}


Eigen::SparseMatrix<double> Drone::getCollisionEnvelope() {
    return collision_envelope;
}


Eigen::MatrixXd Drone::getWaypoints() {
    return waypoints;
}


float Drone::getDeltaT() {
    return config.delta_t;
}


int Drone::getK() {
    return config.K;
}
