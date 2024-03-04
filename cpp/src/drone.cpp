#include <drone.h>

#include <iostream>
#include <cmath>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>


using namespace Eigen;


Drone::Drone(MatrixXd waypoints, MPCConfig config, MPCWeights weights,
            PhysicalLimits limits, SparseDynamics dynamics, 
            VectorXd initial_pos)
    : waypoints(waypoints), initial_pos(initial_pos), config(config),
    weights(weights), limits(limits), dynamics(dynamics),
    collision_envelope(3,3), constSelectionMatrices(config.K)
{   
    std::tie(W, W_dot, W_ddot, W_input) = initBernsteinMatrices(config); // move to struct and constructor 
    std::tie(S_x, S_u, S_x_prime, S_u_prime) = initFullHorizonDynamicsMatrices(dynamics); // move to struct and constructor
    collision_envelope.insert(0,0) = 5.8824; collision_envelope.insert(1,1) = 5.8824; collision_envelope.insert(2,2) = 2.2222; // initialize collision envelope - later move this to a yaml or struct or something

    // Init cost matrices
    quadCost = SparseMatrix<double>(3*(config.n+1),3*(config.n+1));
    quadCost.setZero();
    linearCost = VectorXd(3*(config.n+1));
    linearCost.setZero();
};


Drone::DroneResult Drone::preSolve(const DroneArgs& args) {
    // clear constraints
    nonConstConstraints.clear();

    // extract waypoints in current horizon
    MatrixXd extracted_waypoints = extractWaypointsInCurrentHorizon(args.current_time, waypoints);
    if (extracted_waypoints.size() == 0) {
        throw std::runtime_error("Error: no waypoints within current horizon. Either increase horizon length or add waypoints.");
    }
    
    // extract the penalized steps from the first column of extracted_waypoints
    // note that the first possible penalized step is 1, NOT 0 TODO check this
    VectorXd penalized_steps;
    penalized_steps.resize(extracted_waypoints.rows());
    penalized_steps = extracted_waypoints.block(0,0,extracted_waypoints.rows(),1);

    // initialize optimization parameters
    VariableSelectionMatrices variableSelectionMatrices(config.K, j, penalized_steps);

    // Input smoothness cost
    quadCost += 2 * weights.input_smoothness * (W_ddot.transpose() * W_ddot); // remaining cost terms must be added at solve time

    // Output smoothness cost
    quadCost += 2 * weights.smoothness * W_input.transpose() * S_u_prime.transpose() * M_a.transpose() * M_a * S_u_prime * W_input;
    linearCost += 2 * weights.smoothness * W_input.transpose() * S_u_prime.transpose() * M_a.transpose() * M_a * S_x_prime * x_0;

    // Waypoint position cost and/or equality constraint
    SparseMatrix<double>  G_wp = M_w * M_p * S_u * W_input;
    VectorXd h_wp = extracted_waypoints_pos - M_w * M_p * S_x * x_0;
    quadCost += 2 * weights.waypoint_position * G_wp.transpose() * G_wp;
    std::unique_ptr<Constraint> wpConstraint = std::make_unique<EqualityConstraint>(G_wp, h_wp);
    addConstraint(std::move(wpConstraint), false);

    // Waypoint velocity cost and/or equality constraint
    SparseMatrix<double>  G_wv = M_w * M_v * S_u * W_input;
    VectorXd h_wv = extracted_waypoints_vel - M_w * M_v * S_x * x_0;
    quadCost += 2 * weights.waypoint_velocity * G_wv.transpose() * G_wv;
    std::unique_ptr<Constraint> wvConstraint = std::make_unique<EqualityConstraint>(G_wv, h_wv);
    addConstraint(std::move(wvConstraint), false);

    // Waypoint acceleration cost and/or equality constraint
    SparseMatrix<double>  G_wa = M_w * M_a * S_u * W_input;
    VectorXd h_wa = extracted_waypoints_acc - M_w * M_a * S_x * x_0;
    quadCost += 2 * weights.waypoint_acceleration * G_wa.transpose() * G_wa;
    std::unique_ptr<Constraint> waConstraint = std::make_unique<EqualityConstraint>(G_wa, h_wa);
    addConstraint(std::move(waConstraint), false);

    // Input continuity cost and/or equality constraint

    // Velocity constraint

    std::unique_ptr<Constraint> vConstraint = std::make_unique<PolarInequalityConstraint>(G_v, c_v, -std::numeric_limits<double>::infinity(), limits.v_bar); // FIX CONSTRAINTS NEGATIVE INF
    addConstraint(std::move(vConstraint), false);

    // Acceleration constraint

    std::unique_ptr<Constraint> aConstraint = std::make_unique<PolarInequalityConstraint>(G_a, c_a, -std::numeric_limits<double>::infinity(), limits.a_bar); // FIX CONSTRAINTS NEGATIVE INF
    addConstraint(std::move(aConstraint), false);

    // Collision constraint
    std::unique_ptr<Constraint> cConstraint = std::make_unique<PolarInequalityConstraint>(G_a, c_a, 1.0, inf, bf_gamma);
    addConstraint(std::move(cConstraint), false);

};


MatrixXd Drone::extractWaypointsInCurrentHorizon(const double t,
                                                        const MatrixXd& waypoints) {
    // round all the waypoints to the nearest time step. 
    // negative time steps are allowed -- we will filter them out later
    MatrixXd rounded_waypoints = waypoints;
    rounded_waypoints.col(0) = ((rounded_waypoints.col(0).array() - t) / config.delta_t).round();
    // filter out waypoints that are outside the current horizon
    std::vector<int> rows_in_horizon;
    for (int i = 0; i < rounded_waypoints.rows(); ++i) {
        if (rounded_waypoints(i, 0) >= 1 && rounded_waypoints(i, 0) <= config.K) {
            rows_in_horizon.push_back(i);
        }
    }

    // keep all columns
    VectorXi cols = VectorXi::LinSpaced(rounded_waypoints.cols(), 0,rounded_waypoints.cols());

    // return the rounded waypoints that are within the current horizon. if none, will return empty matrix. error handling elsewhere
    return rounded_waypoints(rows_in_horizon,cols);
};


std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> Drone::initBernsteinMatrices(const MPCConfig& config) {    
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


std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> Drone::initFullHorizonDynamicsMatrices(const SparseDynamics& dynamics) {
    int num_states = dynamics.A.rows();
    int num_inputs = dynamics.B.cols();

    SparseMatrix<double> S_x(num_states*(config.K+1), num_states);
    SparseMatrix<double> S_x_prime(num_states*(config.K+1), num_states);
    SparseMatrix<double> S_u(num_states*(config.K+1), num_inputs*config.K);
    SparseMatrix<double> S_u_prime(num_states*(config.K+1), num_inputs*config.K);

    // Build S_x and S_x_prime --> to do at some point, build A_prime and B_prime from A and B, accounting for identified model time
    SparseMatrix<double> temp_S_x_block(num_states,num_states);
    SparseMatrix<double> temp_S_x_prime_block(num_states,num_states);
    for (int k = 0; k <= config.K; ++k) {
        temp_S_x_block = utils::matrixPower(dynamics.A,k); // necesssary to explicitly make this a sparse matrix to avoid ambiguous call to replaceSparseBlock
        
        if (k == 0) {
            temp_S_x_prime_block = SparseMatrix<double>(num_states,num_states);
        } else {
            temp_S_x_prime_block = dynamics.A_prime * utils::matrixPower(dynamics.A,k-1);
        }
        // temp_S_x_prime_block = dynamics.A_prime * utils::matrixPower(dynamics.A,k);
        utils::replaceSparseBlock(S_x, temp_S_x_block, k * num_states, 0);
        utils::replaceSparseBlock(S_x_prime, temp_S_x_prime_block, k * num_states, 0);
    }

    // Build S_u and S_u_prime
    SparseMatrix<double> S_u_col(num_states * config.K, num_inputs);
    SparseMatrix<double> S_u_prime_col(num_states * config.K, num_inputs);
    SparseMatrix<double> temp_S_u_col_block(num_states,num_inputs);
    SparseMatrix<double> temp_S_u_prime_col_block(num_states,num_inputs);
    for (int k = 0; k < config.K; ++k) {
        temp_S_u_col_block = utils::matrixPower(dynamics.A,k) * dynamics.B;
        utils::replaceSparseBlock(S_u_col, temp_S_u_col_block, k * num_states, 0);
    }

    utils::replaceSparseBlock(S_u_prime_col, dynamics.B_prime, 0, 0);
    for (int k = 1; k < config.K; ++k) {
        temp_S_u_prime_col_block = dynamics.A_prime * utils::matrixPower(dynamics.A,k-1) * dynamics.B;
        utils::replaceSparseBlock(S_u_prime_col, temp_S_u_prime_col_block, k * num_states, 0);
    }

    for (int k = 0; k < config.K; ++k) {
        utils::replaceSparseBlock(S_u, static_cast<SparseMatrix<double>>(S_u_col.block(0, 0, (config.K - k) * num_states, num_inputs)), (k+1) * num_states, k * num_inputs);
        utils::replaceSparseBlock(S_u_prime, static_cast<SparseMatrix<double>>(S_u_prime_col.block(0, 0, (config.K - k) * num_states, num_inputs)), (k+1) * num_states, k * num_inputs);
    }
    return std::make_tuple(S_x, S_u, S_x_prime, S_u_prime);
};


Drone::DroneResult Drone::computeDroneResult(double current_time, VectorXd& zeta_1, VectorXd x_0) {
    DroneResult drone_result;
    
    // input trajectory
    drone_result.control_input_trajectory_vector = W_input * zeta_1;
    drone_result.control_input_trajectory = Map<MatrixXd>(drone_result.control_input_trajectory_vector.data(), 6, config.K).transpose(); // TODO automatically resize based on num inputs

    // state trajectory
    drone_result.state_trajectory_vector = S_x * x_0 + S_u * drone_result.control_input_trajectory_vector;
    drone_result.state_trajectory = Map<MatrixXd>(drone_result.state_trajectory_vector.data(), 6, (config.K+1)).transpose();

    // position trajectory
    drone_result.position_trajectory_vector = constSelectionMatrices.M_p * drone_result.state_trajectory_vector;
    drone_result.position_trajectory = Map<MatrixXd>(drone_result.position_trajectory_vector.data(), 3, (config.K+1)).transpose();

    // Time stamps for position and state trajectories
    drone_result.position_state_time_stamps.resize(config.K+1);
    for (int i = 0; i < config.K+1; ++i) {
        drone_result.position_state_time_stamps(i) = current_time + (i) * config.delta_t;
    }

    // Time stamps for control input trajectory
    drone_result.control_input_time_stamps.resize(config.K);
    for (int i = 0; i < config.K; ++i) {
        drone_result.control_input_time_stamps(i) = current_time + i * config.delta_t;
    }

    drone_result.spline_coeffs = zeta_1;

    return drone_result;
}

void Drone::printUnsatisfiedResiduals(const Residuals& residuals,
                                    SolveOptions& opt) {
    // Helper function to print steps where residuals exceed threshold
    auto printExceedingSteps = [this](const VectorXd& residual, int start, int end, const std::string& message, double threshold, bool wrap = false) {
        std::vector<int> exceedingSteps;
        for (int i = start; i < end; i += 3) { // Iterate in steps of 3
            if (std::abs(residual[i]) > threshold || (i + 1 < end && std::abs(residual[i+1]) > threshold) || (i + 2 < end && std::abs(residual[i+2]) > threshold)) {
                int step = (i - start) / 3 + 1;
                if (wrap) {
                    step = (step - 1) % (config.K+1) + 1; // Wrap step number if it exceeds K
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
        printExceedingSteps(residuals.eq, 0, 3*(config.K+1), "Velocity constraints residual exceeds threshold", opt.eq_threshold);
        printExceedingSteps(residuals.eq, 3*(config.K+1), 6*(config.K+1), "Acceleration constraints residual exceeds threshold", opt.eq_threshold);
        // For collision constraints, wrap the step number if it exceeds K
        printExceedingSteps(residuals.eq, 6*(config.K+1), residuals.eq.size(), "Collision constraints residual exceeds threshold", opt.eq_threshold, true);
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


VectorXd Drone::U_to_zeta_1(const VectorXd& U) {
    SparseQR<SparseMatrix<double>,COLAMDOrdering<int>> solver;
    solver.analyzePattern(W_input);
    solver.factorize(W_input);
    VectorXd zeta_1 = solver.solve(U);
    return zeta_1;
}


VectorXd Drone::getInitialPosition() {
    return initial_pos;
}


SparseMatrix<double> Drone::getCollisionEnvelope() {
    return collision_envelope;
}


MatrixXd Drone::getWaypoints() {
    return waypoints;
}


float Drone::getDeltaT() {
    return config.delta_t;
}


int Drone::getK() {
    return config.K;
}
