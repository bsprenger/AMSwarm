#include "drone.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace amswarm {

// Eigen type aliases
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using Vector3d = Eigen::Vector3d;
template<typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;
template<typename PlainObjectType, int MapOptions, typename StrideType>
using Map = Eigen::Map<PlainObjectType, MapOptions, StrideType>;


Drone::Drone(AMSolverConfig solverConfig, MatrixXd waypoints, MPCConfig mpcConfig, MPCWeights weights,
             PhysicalLimits limits, SparseDynamics dynamics)
    : AMSolver<DroneResult, DroneSolveArgs>(solverConfig), waypoints(waypoints),
    mpcConfig(mpcConfig), weights(weights), limits(limits), dynamics(dynamics),
    collision_envelope(3,3), selectionMats(mpcConfig.K)
{   
    // Initialize Bernstein polynomials and full horizon dynamics matrices
    std::tie(W, W_dot, W_ddot, W_input) = initBernsteinMatrices(mpcConfig); // move to struct and constructor 
    std::tie(S_x, S_u, S_x_prime, S_u_prime) = initFullHorizonDynamicsMatrices(dynamics); // move to struct and constructor
    
    // The inverse of the collision envelope matrix is used in the collision constraints, so we precompute it here
    collision_envelope.insert(0,0) = 1.0 / limits.x_collision_envelope;
    collision_envelope.insert(1,1) = 1.0 / limits.y_collision_envelope;
    collision_envelope.insert(2,2) = 1.0 / limits.z_collision_envelope;

    // Precompute matrices that don't change at solve time
    S_u_W_input = S_u * W_input;
    M_p_S_u_W_input = selectionMats.M_p * S_u_W_input;
    M_v_S_u_W_input = selectionMats.M_v * S_u_W_input;
    M_a_S_u_prime_W_input = selectionMats.M_a * S_u_prime * W_input;
    M_p_S_x = selectionMats.M_p * S_x;
    M_v_S_x = selectionMats.M_v * S_x;
    M_a_S_x_prime = selectionMats.M_a * S_x_prime;

    // Precompute constraint matrices that don't change at solve time
    G_u = SparseMatrix<double>(9,3*(mpcConfig.n+1));
    utils::replaceSparseBlock(G_u, W.block(0,0,3,3*(mpcConfig.n+1)), 0, 0);
    utils::replaceSparseBlock(G_u, W_dot.block(0,0,3,3*(mpcConfig.n+1)), 3, 0);
    utils::replaceSparseBlock(G_u, W_ddot.block(0,0,3,3*(mpcConfig.n+1)), 6, 0);
    G_u_T = G_u.transpose();
    G_u_T_G_u = G_u_T * G_u;
    G_p = SparseMatrix<double>(6 * (mpcConfig.K + 1), 3 * (mpcConfig.n + 1));
    utils::replaceSparseBlock(G_p, M_p_S_u_W_input, 0, 0);
    utils::replaceSparseBlock(G_p, -M_p_S_u_W_input, 3 * (mpcConfig.K + 1), 0);

    // Initialize initial cost matrices (constant cost terms) and constant cost matrices that don't change at solve time
    initialQuadCost = 2 * weights.input_smoothness * (W_ddot.transpose() * W_ddot);
    initialQuadCost += 2 * weights.smoothness * W_input.transpose() * S_u_prime.transpose() * selectionMats.M_a.transpose() * selectionMats.M_a * S_u_prime * W_input;
    initialQuadCost += 2 * weights.input_continuity * G_u_T_G_u;
    initialLinearCost = VectorXd(3*(mpcConfig.n+1));
    linearCostSmoothnessConstTerm = 2 * weights.smoothness * M_a_S_u_prime_W_input.transpose() * M_a_S_x_prime;
};


void Drone::preSolve(const DroneSolveArgs& args) {
    // extract waypoints in current horizon. Each row is a waypoint, where each waypoint is of the form
    // [k, x, y, z, vx, vy, vz, ax, ay, az]. k is the discrete STEP in the current horizon, not the time.
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> extracted_waypoints = extractWaypointsInCurrentHorizon(args.current_time);

    // separate and reshape the waypoints into position, velocity, and acceleration vectors
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
    // note that the first possible penalized step is 1, NOT 0 (since the input cannot affect the initial state so no point in penalizing it)
    VectorXd penalized_steps = extracted_waypoints.block(0,0,extracted_waypoints.rows(),1);

    // Create a matrix that selects the time steps corresponding to the waypoints from either the position, velocity, or acceleration trajectory
    SparseMatrix<double> M_waypoints(3 * penalized_steps.size(), 3 * (mpcConfig.K + 1));
    SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
    for (int i = 0; i < penalized_steps.size(); ++i) {
        utils::replaceSparseBlock(M_waypoints, eye3, 3 * i, 3 * (penalized_steps(i))); // CHECK THIS
    }
    
    // Output smoothness cost
    linearCost += linearCostSmoothnessConstTerm * args.x_0;

    /// --- Add constraints - see thesis document for derivations --- ///
    // Waypoint position cost and/or equality constraint
    SparseMatrix<double>  G_wp = M_waypoints * M_p_S_u_W_input;
    VectorXd h_wp = extracted_waypoints_pos - M_waypoints * M_p_S_x * args.x_0;
    quadCost += 2 * weights.waypoints_pos * G_wp.transpose() * G_wp;
    linearCost += -2 * weights.waypoints_pos * G_wp.transpose() * h_wp;
    if (args.constraintConfig.enable_waypoints_pos_constraint) {
        std::unique_ptr<Constraint> wpConstraint = std::make_unique<EqualityConstraint>(G_wp, h_wp, mpcConfig.waypoints_pos_tol);
        addConstraint(std::move(wpConstraint), false);
    }

    // Waypoint velocity cost and/or equality constraint
    SparseMatrix<double>  G_wv = M_waypoints * M_v_S_u_W_input;
    VectorXd h_wv = extracted_waypoints_vel - M_waypoints * M_v_S_x * args.x_0;
    quadCost += 2 * weights.waypoints_vel * G_wv.transpose() * G_wv;
    linearCost += -2 * weights.waypoints_vel * G_wv.transpose() * h_wv;
    if (args.constraintConfig.enable_waypoints_vel_constraint) {
        std::unique_ptr<Constraint> wvConstraint = std::make_unique<EqualityConstraint>(G_wv, h_wv, mpcConfig.waypoints_vel_tol);
        addConstraint(std::move(wvConstraint), false);
    }

    // Waypoint acceleration cost and/or equality constraint
    SparseMatrix<double>  G_wa = M_waypoints * M_a_S_u_prime_W_input;
    VectorXd h_wa = extracted_waypoints_acc - M_waypoints * M_a_S_x_prime * args.x_0;
    quadCost += 2 * weights.waypoints_acc * G_wa.transpose() * G_wa;
    linearCost += -2 * weights.waypoints_acc * G_wa.transpose() * h_wa;
    if (args.constraintConfig.enable_waypoints_acc_constraint) {
        std::unique_ptr<Constraint> waConstraint = std::make_unique<EqualityConstraint>(G_wa, h_wa, mpcConfig.waypoints_acc_tol);
        addConstraint(std::move(waConstraint), false);
    }

    // Input continuity cost and/or equality constraint
    VectorXd h_u(9);
    h_u << args.u_0, args.u_dot_0, args.u_ddot_0;
    linearCost += -2 * weights.input_continuity * G_u_T * h_u;
    if (args.constraintConfig.enable_input_continuity_constraint) {
        std::unique_ptr<Constraint> uConstraint = std::make_unique<EqualityConstraint>(G_u, h_u, mpcConfig.input_continuity_tol);
        addConstraint(std::move(uConstraint), false);
    }

    // Position constraint
    VectorXd h_p(6 * (mpcConfig.K + 1));
    h_p << limits.p_max.replicate(mpcConfig.K + 1, 1) - M_p_S_x * args.x_0, -limits.p_min.replicate(mpcConfig.K + 1, 1) + M_p_S_x * args.x_0;
    std::unique_ptr<Constraint> pConstraint = std::make_unique<InequalityConstraint>(G_p, h_p, mpcConfig.pos_tol);
    addConstraint(std::move(pConstraint), false);

    // Velocity constraint
    VectorXd c_v = M_v_S_x * args.x_0;
    std::unique_ptr<Constraint> vConstraint = std::make_unique<PolarInequalityConstraint>(M_v_S_u_W_input, c_v, -std::numeric_limits<double>::infinity(), limits.v_bar, 1.0, mpcConfig.vel_tol);
    addConstraint(std::move(vConstraint), false);

    // Acceleration constraint
    VectorXd c_a = M_a_S_x_prime * args.x_0;
    std::unique_ptr<Constraint> aConstraint = std::make_unique<PolarInequalityConstraint>(M_a_S_u_prime_W_input, c_a, -std::numeric_limits<double>::infinity(), limits.a_bar, 1.0, mpcConfig.acc_tol);
    addConstraint(std::move(aConstraint), false);

    // Collision constraints
    for (int i = 0; i < args.num_obstacles; ++i) {
        SparseMatrix<double> G_c = args.obstacle_envelopes[i] * M_p_S_u_W_input;
        VectorXd c_c = args.obstacle_envelopes[i] * (M_p_S_x * args.x_0 - args.obstacle_positions[i]);
        std::unique_ptr<Constraint> cConstraint = std::make_unique<PolarInequalityConstraint>(G_c, c_c, 1.0, std::numeric_limits<double>::infinity(), mpcConfig.bf_gamma, mpcConfig.collision_tol);
        addConstraint(std::move(cConstraint), false);
    }
};

DroneResult Drone::postSolve(const VectorXd& zeta, const DroneSolveArgs& args) {
    DroneResult drone_result;

    // get state trajectory vector from spline coefficients, reshape it into a matrix where each row is the state at a time step
    drone_result.state_trajectory_vector = S_x * args.x_0 + S_u_W_input * zeta;
    drone_result.state_trajectory = Eigen::Map<MatrixXd>(drone_result.state_trajectory_vector.data(), 6, (mpcConfig.K+1)).transpose();

    // extract position trajectory from state trajectory, reshape it into a matrix where each row is the position at a time step
    drone_result.position_trajectory_vector = selectionMats.M_p * drone_result.state_trajectory_vector;
    drone_result.position_trajectory = Eigen::Map<MatrixXd>(drone_result.position_trajectory_vector.data(), 3, (mpcConfig.K+1)).transpose();

    // get input position reference from spline coefficients, reshape it into a matrix where each row is the input position at a time step
    drone_result.input_position_trajectory_vector = W * zeta;
    drone_result.input_position_trajectory = Eigen::Map<MatrixXd>(drone_result.input_position_trajectory_vector.data(), 3, (mpcConfig.K)).transpose();

    // get input velocity reference from spline coefficients, reshape it into a matrix where each row is the input velocity at a time step
    drone_result.input_velocity_trajectory_vector = W_dot * zeta;
    drone_result.input_velocity_trajectory = Eigen::Map<MatrixXd>(drone_result.input_velocity_trajectory_vector.data(), 3, (mpcConfig.K)).transpose();

    // get input acceleration reference from spline coefficients, reshape it into a matrix where each row is the input acceleration at a time step
    drone_result.input_acceleration_trajectory_vector = W_ddot * zeta;
    drone_result.input_acceleration_trajectory = Eigen::Map<MatrixXd>(drone_result.input_acceleration_trajectory_vector.data(), 3, (mpcConfig.K)).transpose();

    drone_result.spline_coeffs = zeta; // spline coeffs directly from optimization results

    return drone_result;
};


Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Drone::extractWaypointsInCurrentHorizon(double t) {
    // copy the waypoints
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rounded_waypoints = waypoints;

    // round the first column of the waypoints to the nearest discrete time step for the given frequency, relative to the current time
    // note that at this point there could be negative time steps for waypoints that happened before the current time
    rounded_waypoints.col(0) = ((rounded_waypoints.col(0).array() - t) / (1 / mpcConfig.mpc_freq)).round();
    
    // find the time steps with waypoints that are within the current horizon
    // note that the smallest time step allowed is 1, not 0: the input cannot affect the initial state so we do not want to extract it
    std::vector<int> rows_in_horizon;
    for (int i = 0; i < rounded_waypoints.rows(); ++i) {
        if (rounded_waypoints(i, 0) >= 1 && rounded_waypoints(i, 0) <= mpcConfig.K) {
            rows_in_horizon.push_back(i);
        }
    }

    if (rows_in_horizon.empty()) {
        throw std::runtime_error("Error: no waypoints within current horizon. Either increase horizon length or add waypoints.");
    }

    // Create a matrix to hold filtered waypoints
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> filtered_waypoints(rows_in_horizon.size(), rounded_waypoints.cols());

    // Copy only the rows in the horizon to the new matrix
    for (size_t i = 0; i < rows_in_horizon.size(); ++i) {
        filtered_waypoints.row(i) = rounded_waypoints.row(rows_in_horizon[i]);
    }

    // return the rounded waypoints that are within the current horizon
    return filtered_waypoints;
};


std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> Drone::initBernsteinMatrices(const MPCConfig& mpcConfig) {    
    // See thesis document for derivation of these matrices
    SparseMatrix<double> W(3*mpcConfig.K,3*(mpcConfig.n+1));
    SparseMatrix<double> W_dot(3*mpcConfig.K,3*(mpcConfig.n+1));
    SparseMatrix<double> W_ddot(3*mpcConfig.K,3*(mpcConfig.n+1));
    SparseMatrix<double> W_input(6*mpcConfig.K,3*(mpcConfig.n+1)); // The drones take the position and velocity references as input. This matrix is used to construct the inputs from the spline coefficients

    double t_f = (1 / mpcConfig.mpc_freq)*(mpcConfig.K-1);
    float t;
    float val;
    float dot_val;
    float dotdot_val;

    for (int k=0;k < mpcConfig.K;k++) { 
        t  = k*(1 / mpcConfig.mpc_freq);
        float t_f_minus_t = t_f - t;
        float t_pow_n = pow(t_f,mpcConfig.n);
        for (int m=0;m<mpcConfig.n+1;m++) {
            val = pow(t,m)*utils::nchoosek(mpcConfig.n,m)*pow(t_f_minus_t, mpcConfig.n-m)/t_pow_n;

            if (k == 0 && m == 0){
                dot_val = -mpcConfig.n*pow(t_f,-1);
            } else if (k == mpcConfig.K-1 && m == mpcConfig.n) {
                dot_val = mpcConfig.n*pow(t_f,-1);
            } else {
                dot_val = pow(t_f,-mpcConfig.n)*utils::nchoosek(mpcConfig.n,m)*(m*pow(t,m-1)*pow(t_f-t,mpcConfig.n-m) - pow(t,m)*(mpcConfig.n-m)*pow(t_f-t,mpcConfig.n-m-1));
            }

            if (k == 0 && m == 0) {
                dotdot_val = mpcConfig.n*(mpcConfig.n-1)*pow(t_f,-2);
            } else if (k == mpcConfig.K-1 && m == mpcConfig.n) {
                dotdot_val = mpcConfig.n*(mpcConfig.n-1)*pow(t_f,-2);
            } else if (k == 0 && m == 1) {
                dotdot_val = -2*mpcConfig.n*(mpcConfig.n-1)*pow(t_f,-2);
            } else if (k == mpcConfig.K-1 && m == mpcConfig.n-1) {
                dotdot_val = -2*mpcConfig.n*(mpcConfig.n-1)*pow(t_f,-2);
            } else {
                dotdot_val = pow(t_f,-mpcConfig.n)*utils::nchoosek(mpcConfig.n,m)*(
                    m*(m-1)*pow(t,m-2)*pow(t_f-t,mpcConfig.n-m)
                    -2*m*(mpcConfig.n-m)*pow(t,m-1)*pow(t_f-t,mpcConfig.n-m-1)
                    +(mpcConfig.n-m)*(mpcConfig.n-m-1)*pow(t,m)*pow(t_f-t,mpcConfig.n-m-2));
            }

            if (val != 0) { // don't bother filling in the value if zero - we are using sparse matrix
                W.coeffRef(3 * k, m) = val;
                W.coeffRef(3 * k + 1, m + (mpcConfig.n + 1)) = val;
                W.coeffRef(3 * k + 2, m + 2 * (mpcConfig.n + 1)) = val;
            }
            if (dot_val != 0) {
                W_dot.coeffRef(3 * k, m) = dot_val;
                W_dot.coeffRef(3 * k + 1, m + (mpcConfig.n + 1)) = dot_val;
                W_dot.coeffRef(3 * k + 2, m + 2 * (mpcConfig.n + 1)) = dot_val;
            }
            if (dotdot_val != 0) {
                W_ddot.coeffRef(3 * k, m) = dotdot_val;
                W_ddot.coeffRef(3 * k + 1, m + (mpcConfig.n + 1)) = dotdot_val;
                W_ddot.coeffRef(3 * k + 2, m + 2 * (mpcConfig.n + 1)) = dotdot_val;
            }
        }
    }
    
    // construct input matrix
    for (int block = 0; block < mpcConfig.K; ++block) {
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
    // see thesis document for derivation of these matrices
    int num_states = dynamics.A.rows();
    int num_inputs = dynamics.B.cols();

    SparseMatrix<double> S_x(num_states*(mpcConfig.K+1), num_states);
    SparseMatrix<double> S_x_prime(num_states*(mpcConfig.K+1), num_states);
    SparseMatrix<double> S_u(num_states*(mpcConfig.K+1), num_inputs*mpcConfig.K);
    SparseMatrix<double> S_u_prime(num_states*(mpcConfig.K+1), num_inputs*mpcConfig.K);

    // Build S_x and S_x_prime --> to do at some point, build A_prime and B_prime from A and B, accounting for identified model time
    SparseMatrix<double> temp_S_x_block(num_states,num_states);
    SparseMatrix<double> temp_S_x_prime_block(num_states,num_states);
    for (int k = 0; k <= mpcConfig.K; ++k) {
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
    SparseMatrix<double> S_u_col(num_states * mpcConfig.K, num_inputs);
    SparseMatrix<double> S_u_prime_col(num_states * mpcConfig.K, num_inputs);
    SparseMatrix<double> temp_S_u_col_block(num_states,num_inputs);
    SparseMatrix<double> temp_S_u_prime_col_block(num_states,num_inputs);
    for (int k = 0; k < mpcConfig.K; ++k) {
        temp_S_u_col_block = utils::matrixPower(dynamics.A,k) * dynamics.B;
        utils::replaceSparseBlock(S_u_col, temp_S_u_col_block, k * num_states, 0);
    }

    utils::replaceSparseBlock(S_u_prime_col, dynamics.B_prime, 0, 0);
    for (int k = 1; k < mpcConfig.K; ++k) {
        temp_S_u_prime_col_block = dynamics.A_prime * utils::matrixPower(dynamics.A,k-1) * dynamics.B;
        utils::replaceSparseBlock(S_u_prime_col, temp_S_u_prime_col_block, k * num_states, 0);
    }

    for (int k = 0; k < mpcConfig.K; ++k) {
        utils::replaceSparseBlock(S_u, static_cast<SparseMatrix<double>>(S_u_col.block(0, 0, (mpcConfig.K - k) * num_states, num_inputs)), (k+1) * num_states, k * num_inputs);
        utils::replaceSparseBlock(S_u_prime, static_cast<SparseMatrix<double>>(S_u_prime_col.block(0, 0, (mpcConfig.K - k) * num_states, num_inputs)), (k+1) * num_states, k * num_inputs);
    }
    return std::make_tuple(S_x, S_u, S_x_prime, S_u_prime);
};

void DroneResult::advanceForNextSolveStep() {
    // See explanation in header file
    // Shift everything up by 3 elements (to get rid of the first time step)
    // then, extrapolate a new position on the end as an estimate for the last time step
    // This will be used by other drones to avoid collisions
    int size = position_trajectory_vector.size();
    position_trajectory_vector.segment(0, size - 3) = position_trajectory_vector.segment(3, size - 3);
    Vector3d extrapolated_position = position_trajectory_vector.tail(3) + (position_trajectory_vector.tail(3) - position_trajectory_vector.segment(size - 6, 3));
    position_trajectory_vector.tail(3) = extrapolated_position;
    position_trajectory = Eigen::Map<MatrixXd>(position_trajectory_vector.data(), 3, position_trajectory_vector.size() / 3).transpose();

    // Advance the input trajectories - no need to extrapolate as we only check the first row, kinda hacky - could be better
    int inputSize = input_position_trajectory_vector.size();
    input_position_trajectory_vector.segment(0, inputSize - 3) = input_position_trajectory_vector.segment(3, inputSize - 3);
    input_velocity_trajectory_vector.segment(0, inputSize - 3) = input_velocity_trajectory_vector.segment(3, inputSize - 3);
    input_acceleration_trajectory_vector.segment(0, inputSize - 3) = input_acceleration_trajectory_vector.segment(3, inputSize - 3);
    input_position_trajectory = Eigen::Map<MatrixXd>(input_position_trajectory_vector.data(), 3, (input_position_trajectory_vector.size()/3)).transpose();
    input_velocity_trajectory = Eigen::Map<MatrixXd>(input_velocity_trajectory_vector.data(), 3, (input_velocity_trajectory_vector.size()/3)).transpose();
    input_acceleration_trajectory = Eigen::Map<MatrixXd>(input_acceleration_trajectory_vector.data(), 3, (input_acceleration_trajectory_vector.size()/3)).transpose();
}

DroneResult DroneResult::generateInitialDroneResult(const VectorXd& initial_position, int K) {
    // See explanation in header file
    DroneResult drone_result;

    // generate state trajectory by appending zero velocity to initial position and replicating
    VectorXd initial_state = VectorXd::Zero(6);
    initial_state.head(3) = initial_position;
    drone_result.state_trajectory_vector = initial_state.replicate(K+1,1);
    drone_result.state_trajectory = Eigen::Map<MatrixXd>(drone_result.state_trajectory_vector.data(), 6, (K+1)).transpose();

    // generate position trajectory by replicating initial position
    drone_result.position_trajectory_vector = initial_position.replicate(K+1,1);
    drone_result.position_trajectory = Eigen::Map<MatrixXd>(drone_result.position_trajectory_vector.data(), 3, (K+1)).transpose();

    // generate input position trajectory by replicating initial position K times
    drone_result.input_position_trajectory_vector = initial_position.replicate(K,1);
    drone_result.input_position_trajectory = Eigen::Map<MatrixXd>(drone_result.input_position_trajectory_vector.data(), 3, (K)).transpose();

    // generate input velocity trajectory by replicating zero K times
    drone_result.input_velocity_trajectory_vector = VectorXd::Zero(3*K);
    drone_result.input_velocity_trajectory = Eigen::Map<MatrixXd>(drone_result.input_velocity_trajectory_vector.data(), 3, (K)).transpose();

    // generate input acceleration trajectory by replicating zero K times
    drone_result.input_acceleration_trajectory_vector = VectorXd::Zero(3*K);
    drone_result.input_acceleration_trajectory = Eigen::Map<MatrixXd>(drone_result.input_acceleration_trajectory_vector.data(), 3, (K)).transpose();

    return drone_result;
}

SparseMatrix<double> Drone::getCollisionEnvelope() {
    return collision_envelope;
}


int Drone::getK() {
    return mpcConfig.K;
}

} // namespace amswarm
