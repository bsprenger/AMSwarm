#include <drone.h>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>


using namespace Eigen;


Drone::Drone(UpdateMethod method, MatrixXd waypoints, MPCConfig config, MPCWeights weights,
            PhysicalLimits limits, SparseDynamics dynamics)
    : AMSolver<DroneResult, DroneSolveArgs>(method), waypoints(waypoints),
    config(config), weights(weights), limits(limits), dynamics(dynamics),
    collision_envelope(3,3), selectionMats(config.K)
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


void Drone::preSolve(const DroneSolveArgs& args) {
    // extract waypoints in current horizon
    // first column is the STEP, not the TIME TODO check this and also round on init instead of each time
    // Create a matrix to hold filtered waypoints
    Matrix<double, Dynamic, Dynamic, RowMajor> extracted_waypoints = extractWaypointsInCurrentHorizon(args.current_time);

    int n = extracted_waypoints.rows();
    VectorXd extracted_waypoints_pos(3 * n);
    VectorXd extracted_waypoints_vel(3 * n);
    VectorXd extracted_waypoints_acc(3 * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 3; ++j) {
            extracted_waypoints_pos(i * 3 + j) = extracted_waypoints(i, 1 + j);
            extracted_waypoints_vel(i * 3 + j) = extracted_waypoints(i, 4 + j);
            extracted_waypoints_acc(i * 3 + j) = extracted_waypoints(i, 7 + j);
        }
    }
    
    // extract the penalized steps from the first column of extracted_waypoints
    // note that the first possible penalized step is 1, NOT 0 TODO check this
    VectorXd penalized_steps;
    penalized_steps.resize(extracted_waypoints.rows());
    penalized_steps = extracted_waypoints.block(0,0,extracted_waypoints.rows(),1);

    // initialize optimization parameters
    SparseMatrix<double> M_waypoints(3 * penalized_steps.size(), 3 * (config.K + 1));
    SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
    for (int i = 0; i < penalized_steps.size(); ++i) {
        utils::replaceSparseBlock(M_waypoints, eye3, 3 * i, 3 * (penalized_steps(i))); // CHECK THIS
    }

    // Input smoothness cost
    quadCost += 2 * weights.input_smoothness * (W_ddot.transpose() * W_ddot); // remaining cost terms must be added at solve time

    // Output smoothness cost
    quadCost += 2 * weights.smoothness * W_input.transpose() * S_u_prime.transpose() * selectionMats.M_a.transpose() * selectionMats.M_a * S_u_prime * W_input;
    linearCost += 2 * weights.smoothness * W_input.transpose() * S_u_prime.transpose() * selectionMats.M_a.transpose() * selectionMats.M_a * S_x_prime * args.x_0;

    // Waypoint position cost and/or equality constraint
    SparseMatrix<double>  G_wp = M_waypoints * selectionMats.M_p * S_u * W_input;
    VectorXd h_wp = extracted_waypoints_pos - M_waypoints * selectionMats.M_p * S_x * args.x_0;
    quadCost += 2 * weights.waypoints_pos * G_wp.transpose() * G_wp;
    linearCost += -2 * weights.waypoints_pos * G_wp.transpose() * h_wp;
    if (args.constraintConfig.enable_waypoints_pos_constraint) {
        std::unique_ptr<Constraint> wpConstraint = std::make_unique<EqualityConstraint>(G_wp, h_wp);
        addConstraint(std::move(wpConstraint), false);
    }

    // Waypoint velocity cost and/or equality constraint
    SparseMatrix<double>  G_wv = M_waypoints * selectionMats.M_v * S_u * W_input;
    VectorXd h_wv = extracted_waypoints_vel - M_waypoints * selectionMats.M_v * S_x * args.x_0;
    quadCost += 2 * weights.waypoints_vel * G_wv.transpose() * G_wv;
    linearCost += -2 * weights.waypoints_vel * G_wv.transpose() * h_wv;
    if (args.constraintConfig.enable_waypoints_vel_constraint) {
        std::unique_ptr<Constraint> wvConstraint = std::make_unique<EqualityConstraint>(G_wv, h_wv);
        addConstraint(std::move(wvConstraint), false);
    }

    // Waypoint acceleration cost and/or equality constraint
    SparseMatrix<double>  G_wa = M_waypoints * selectionMats.M_a * S_u_prime * W_input;
    VectorXd h_wa = extracted_waypoints_acc - M_waypoints * selectionMats.M_a * S_x_prime * args.x_0;
    quadCost += 2 * weights.waypoints_acc * G_wa.transpose() * G_wa;
    linearCost += -2 * weights.waypoints_acc * G_wa.transpose() * h_wa;
    if (args.constraintConfig.enable_waypoints_acc_constraint) {
        std::unique_ptr<Constraint> waConstraint = std::make_unique<EqualityConstraint>(G_wa, h_wa);
        addConstraint(std::move(waConstraint), false);
    }

    // Input continuity cost and/or equality constraint
    SparseMatrix<double> G_u(9,3*(config.n+1));
    SparseMatrix<double> W_block = W.block(0,0,3,3*(config.n+1)); // necessary to explicitly make this a sparse matrix to avoid ambiguous call to replaceSparseBlock. TODO fix later
    SparseMatrix<double> W_dot_block = W_dot.block(0,0,3,3*(config.n+1));
    SparseMatrix<double> W_ddot_block = W_ddot.block(0,0,3,3*(config.n+1));
    utils::replaceSparseBlock(G_u, W_block, 0, 0);
    utils::replaceSparseBlock(G_u, W_dot_block, 3, 0);
    utils::replaceSparseBlock(G_u, W_ddot_block, 6, 0);
    VectorXd h_u(9);
    h_u << args.u_0, args.u_dot_0, args.u_ddot_0;
    quadCost += 2 * weights.input_continuity * G_u.transpose() * G_u;
    linearCost += -2 * weights.input_continuity * G_u.transpose() * h_u;
    if (args.constraintConfig.enable_input_continuity_constraint) {
        std::unique_ptr<Constraint> uConstraint = std::make_unique<EqualityConstraint>(G_u, h_u);
        addConstraint(std::move(uConstraint), false);
    }

    // Position constraint
    SparseMatrix<double> G_p(6 * (config.K + 1), 3 * (config.n + 1));
    SparseMatrix<double> G_p_block1 = selectionMats.M_p * S_u * W_input;
    SparseMatrix<double> G_p_block2 = -selectionMats.M_p * S_u * W_input;
    utils::replaceSparseBlock(G_p, G_p_block1, 0, 0);
    utils::replaceSparseBlock(G_p, G_p_block2, 3 * (config.K + 1), 0);
    VectorXd h_p(6 * (config.K + 1));
    h_p << limits.p_max.replicate(config.K + 1, 1) - selectionMats.M_p * S_x * args.x_0, -limits.p_min.replicate(config.K + 1, 1) + selectionMats.M_p * S_x * args.x_0;
    std::unique_ptr<Constraint> pConstraint = std::make_unique<InequalityConstraint>(G_p, h_p);
    addConstraint(std::move(pConstraint), false);

    // Velocity constraint
    SparseMatrix<double> G_v = selectionMats.M_v * S_u * W_input;
    VectorXd c_v = selectionMats.M_v * S_x * args.x_0;
    std::unique_ptr<Constraint> vConstraint = std::make_unique<PolarInequalityConstraint>(G_v, c_v, -std::numeric_limits<double>::infinity(), limits.v_bar, 1.0);
    addConstraint(std::move(vConstraint), false);

    // Acceleration constraint
    SparseMatrix<double> G_a = selectionMats.M_a * S_u_prime * W_input;
    VectorXd c_a = selectionMats.M_a * S_x_prime * args.x_0;
    std::unique_ptr<Constraint> aConstraint = std::make_unique<PolarInequalityConstraint>(G_a, c_a, -std::numeric_limits<double>::infinity(), limits.a_bar, 1.0);
    addConstraint(std::move(aConstraint), false);

    // Collision constraints
    for (int i = 0; i < args.num_obstacles; ++i) {
        SparseMatrix<double> G_c = args.obstacle_envelopes[i] * selectionMats.M_p * S_u * W_input;
        VectorXd c_c = args.obstacle_envelopes[i] * (selectionMats.M_p * S_x * args.x_0 - args.obstacle_positions[i]);
        std::unique_ptr<Constraint> cConstraint = std::make_unique<PolarInequalityConstraint>(G_c, c_c, 1.0, std::numeric_limits<double>::infinity(), config.bf_gamma);
        addConstraint(std::move(cConstraint), false);
    }
};

DroneResult Drone::postSolve(const VectorXd& zeta, const DroneSolveArgs& args) {
    DroneResult drone_result;

    // state trajectory
    drone_result.state_trajectory_vector = S_x * args.x_0 + S_u * W_input * zeta;
    drone_result.state_trajectory = Map<MatrixXd>(drone_result.state_trajectory_vector.data(), 6, (config.K+1)).transpose();

    // position trajectory
    drone_result.position_trajectory_vector = selectionMats.M_p * drone_result.state_trajectory_vector;
    drone_result.position_trajectory = Map<MatrixXd>(drone_result.position_trajectory_vector.data(), 3, (config.K+1)).transpose();

    // input position trajectory
    drone_result.input_position_trajectory_vector = W * zeta;
    drone_result.input_position_trajectory = Map<MatrixXd>(drone_result.input_position_trajectory_vector.data(), 3, (config.K)).transpose();

    // input velocity trajectory
    drone_result.input_velocity_trajectory_vector = W_dot * zeta;
    drone_result.input_velocity_trajectory = Map<MatrixXd>(drone_result.input_velocity_trajectory_vector.data(), 3, (config.K)).transpose();

    // input acceleration trajectory
    drone_result.input_acceleration_trajectory_vector = W_ddot * zeta;
    drone_result.input_acceleration_trajectory = Map<MatrixXd>(drone_result.input_acceleration_trajectory_vector.data(), 3, (config.K)).transpose();

    // Spline
    drone_result.spline_coeffs = zeta;

    return drone_result;
};


Matrix<double, Dynamic, Dynamic, RowMajor> Drone::extractWaypointsInCurrentHorizon(double t) {
    // round all the waypoints to the nearest time step. 
    // negative time steps are allowed -- we will filter them out later
    Matrix<double, Dynamic, Dynamic, RowMajor> rounded_waypoints = waypoints;
    rounded_waypoints.col(0) = ((rounded_waypoints.col(0).array() - t) / (1 / config.mpc_freq)).round();
    // filter out waypoints that are outside the current horizon
    std::vector<int> rows_in_horizon;
    for (int i = 0; i < rounded_waypoints.rows(); ++i) {
        if (rounded_waypoints(i, 0) >= 1 && rounded_waypoints(i, 0) <= config.K) {
            rows_in_horizon.push_back(i);
        }
    }

    if (rows_in_horizon.empty()) {
        throw std::runtime_error("Error: no waypoints within current horizon. Either increase horizon length or add waypoints.");
    }

    // Create a matrix to hold filtered waypoints
    Matrix<double, Dynamic, Dynamic, RowMajor> filtered_waypoints(rows_in_horizon.size(), rounded_waypoints.cols());

    // Copy the rows in the horizon to the new matrix
    for (size_t i = 0; i < rows_in_horizon.size(); ++i) {
        filtered_waypoints.row(i) = rounded_waypoints.row(rows_in_horizon[i]);
    }

    // return the rounded waypoints that are within the current horizon
    return filtered_waypoints;
};


std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> Drone::initBernsteinMatrices(const MPCConfig& config) {    
    SparseMatrix<double> W(3*config.K,3*(config.n+1));
    SparseMatrix<double> W_dot(3*config.K,3*(config.n+1));
    SparseMatrix<double> W_ddot(3*config.K,3*(config.n+1));
    SparseMatrix<double> W_input(6*config.K,3*(config.n+1));

    double t_f = (1 / config.mpc_freq)*(config.K-1);
    float t;
    float val;
    float dot_val;
    float dotdot_val;

    for (int k=0;k < config.K;k++) { 
        t  = k*(1 / config.mpc_freq);
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

void DroneResult::advanceForNextSolveStep() {
    // Shift everything up by 3 elements and then extrapolate the new last position
    int size = position_trajectory_vector.size();
    position_trajectory_vector.segment(0, size - 3) = position_trajectory_vector.segment(3, size - 3);
    Vector3d extrapolated_position = position_trajectory_vector.tail(3) + (position_trajectory_vector.tail(3) - position_trajectory_vector.segment(size - 6, 3));
    position_trajectory_vector.tail(3) = extrapolated_position;
    position_trajectory = Map<MatrixXd>(position_trajectory_vector.data(), 3, position_trajectory_vector.size() / 3).transpose();

    // Advance the input trajectories - no need to extrapolate as we only check the first row TODO add extrapolate for future just in case
    int inputSize = input_position_trajectory_vector.size();
    input_position_trajectory_vector.segment(0, inputSize - 3) = input_position_trajectory_vector.segment(3, inputSize - 3);
    input_velocity_trajectory_vector.segment(0, inputSize - 3) = input_velocity_trajectory_vector.segment(3, inputSize - 3);
    input_acceleration_trajectory_vector.segment(0, inputSize - 3) = input_acceleration_trajectory_vector.segment(3, inputSize - 3);
    input_position_trajectory = Map<MatrixXd>(input_position_trajectory_vector.data(), 3, (input_position_trajectory_vector.size()/3)).transpose();
    input_velocity_trajectory = Map<MatrixXd>(input_velocity_trajectory_vector.data(), 3, (input_velocity_trajectory_vector.size()/3)).transpose();
    input_acceleration_trajectory = Map<MatrixXd>(input_acceleration_trajectory_vector.data(), 3, (input_acceleration_trajectory_vector.size()/3)).transpose();
}

DroneResult DroneResult::generateInitialDroneResult(const VectorXd& initial_position, int K) {
    DroneResult drone_result;

    // generate state trajectory by appending zero velocity to initial position and replicating
    VectorXd initial_state = VectorXd::Zero(6);
    initial_state.head(3) = initial_position;
    drone_result.state_trajectory_vector = initial_state.replicate(K+1,1);
    drone_result.state_trajectory = Map<MatrixXd>(drone_result.state_trajectory_vector.data(), 6, (K+1)).transpose();

    // generate position trajectory by replicating initial position
    drone_result.position_trajectory_vector = initial_position.replicate(K+1,1);
    drone_result.position_trajectory = Map<MatrixXd>(drone_result.position_trajectory_vector.data(), 3, (K+1)).transpose();

    // generate input position trajectory by replicating initial position K times
    drone_result.input_position_trajectory_vector = initial_position.replicate(K,1);
    drone_result.input_position_trajectory = Map<MatrixXd>(drone_result.input_position_trajectory_vector.data(), 3, (K)).transpose();

    // generate input velocity trajectory by replicating zero K times
    drone_result.input_velocity_trajectory_vector = VectorXd::Zero(3*K);
    drone_result.input_velocity_trajectory = Map<MatrixXd>(drone_result.input_velocity_trajectory_vector.data(), 3, (K)).transpose();

    // generate input acceleration trajectory by replicating zero K times
    drone_result.input_acceleration_trajectory_vector = VectorXd::Zero(3*K);
    drone_result.input_acceleration_trajectory = Map<MatrixXd>(drone_result.input_acceleration_trajectory_vector.data(), 3, (K)).transpose();

    return drone_result;
}

SparseMatrix<double> Drone::getCollisionEnvelope() {
    return collision_envelope;
}


MatrixXd Drone::getWaypoints() {
    return waypoints;
}


int Drone::getK() {
    return config.K;
}
