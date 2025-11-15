#ifndef DRONE_H
#define DRONE_H

#include "amsolver.h"
#include "utils.h"

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

/**
 * @file drone.h
 * 
 * This file contains the Drone class, which is a subclass of AMSolver.
 * The Drone class is used to solve the drone trajectory optimization problem.
 * The drone trajectory optimization problem is a model predictive control (MPC) problem
 * that aims to find the optimal trajectory for a drone to follow given a set of waypoints.
 * The drone must follow the waypoints while avoiding obstacles and satisfying physical constraints.
 * 
 * We define the following interface structs which are used to pass arguments to the Drone class functions
 * and to return results from the Drone class functions to the user's code:
 * - DroneResult: Structure to hold the results of the drone trajectory optimization.
 * - ConstraintConfig: Structure to hold the configuration for the constraints.
 * - DroneSolveArgs: Structure to hold the arguments for the drone solve function.
 * 
 * We define the following structs which are used to simplify construction:
 * - MPCWeights: Structure to hold the weights for the MPC cost function.
 * - MPCConfig: Structure to hold the configuration for the MPC solver.
 * - PhysicalLimits: Structure to hold the physical limits for the drone.
 * - SparseDynamics: Structure to hold the sparse dynamics matrices for the drone.
 */

namespace amswarm {

/**
 * @struct DroneResult
 * Structure to hold the results of the drone trajectory optimization.
 * Includes the position, state, and input trajectories along with the spline coefficients.
 */
struct DroneResult {
    Eigen::MatrixXd position_trajectory; // Trajectory MATRIX. Each row is the position vector at a time step. Dimensions: K+1 x 3
    Eigen::VectorXd position_trajectory_vector; // Reshape of the above. Dimensions: 3(K+1) x 1, each 3-element segment is the position at a time step
    Eigen::MatrixXd state_trajectory; // State MATRIX. Drone state is [position, velocity]. Each row is the state vector at a time step. Dimensions: K+1 x 6
    Eigen::VectorXd state_trajectory_vector; // Reshape of the above. Dimensions: 6(K+1) x 1, each 6-element segment is the state at a time step
    Eigen::MatrixXd input_position_trajectory; // Input position reference MATRIX. Each row is the input position reference at a time step. Dimensions: K x 3
    Eigen::VectorXd input_position_trajectory_vector; // Reshape of the above. Dimensions: 3K x 1, each 3-element segment is the input position reference at a time step
    Eigen::MatrixXd input_velocity_trajectory; // Input velocity reference MATRIX (derivative of input position reference). Each row is the input velocity reference at a time step. Dimensions: K x 3
    Eigen::VectorXd input_velocity_trajectory_vector; // Reshape of the above. Dimensions: 3K x 1, each 3-element segment is the input velocity reference at a time step
    Eigen::MatrixXd input_acceleration_trajectory; // Input acceleration reference MATRIX (derivative of input velocity reference). Each row is the input acceleration reference at a time step. Dimensions: K x 3
    Eigen::VectorXd input_acceleration_trajectory_vector; // Reshape of the above. Dimensions: 3K x 1, each 3-element segment is the input acceleration reference at a time step
    Eigen::VectorXd spline_coeffs; // Input is parameterized as a spline during the optimization. These are the spline coefficients. The above input trajectories are generated from these coefficients

    /**
     * @brief Advances the drone's state and input trajectories by one step.
     * This method shifts the entire trajectory (state, position, inputs) up by one step, discarding the first
     * step and estimating a new last step of the position trajectory by extrapolating.
     * At each solve step, each drone needs the other drones' predicted trajectories, starting from the current
     * time step. This method is used to prepare the drone's trajectory for the next solve step so that the
     * other drones can use this struct for the predicted trajectory.
     */
    void advanceForNextSolveStep();

    /**
     * @brief Generates an initial `DroneResult` object for a drone.
     * This method sets the drone's initial state trajectory to consist of the initial position
     * with zero velocities for the whole horizon (i.e. stationary at initial position).
     * The input position trajectory is set to the initial position for the whole horizon, which
     * is critical for the first optimization solve step if input continuity constraints are enabled,
     * so that the first input is not an abrupt change from the drone's current state.
     * 
     * @param initial_position The drone's initial position as a Eigen::VectorXd.
     * @param K The horizon length over which to initialize the trajectories.
     * @return A `DroneResult` object initialized with the drone's initial state and input trajectories.
     */
    static DroneResult generateInitialDroneResult(const Eigen::VectorXd& initial_position, int K);
};

/**
 * @struct ConstraintConfig
 * @brief Configuration for enabling or disabling various constraints in the optimization problem.
 */
struct ConstraintConfig {
    bool enable_waypoints_pos_constraint = true;
    bool enable_waypoints_vel_constraint = true;
    bool enable_waypoints_acc_constraint = true;
    bool enable_input_continuity_constraint = true;

    void setWaypointsConstraints(bool pos, bool vel, bool acc) {
        enable_waypoints_pos_constraint = pos;
        enable_waypoints_vel_constraint = vel;
        enable_waypoints_acc_constraint = acc;
    }

    void setInputContinuityConstraints(bool flag) {
        enable_input_continuity_constraint = flag;
    }
};

/**
 * @struct DroneSolveArgs
 * @brief Arguments required for solving the drone trajectory optimization problem.
 */
struct DroneSolveArgs {
    double current_time = 0.0;
    int num_obstacles = 0;
    std::vector<Eigen::SparseMatrix<double>> obstacle_envelopes = {};
    std::vector<Eigen::VectorXd> obstacle_positions = {};
    Eigen::VectorXd x_0 = Eigen::VectorXd::Zero(6); // Initial state [position, velocity]
    Eigen::VectorXd u_0 = Eigen::VectorXd::Zero(3); // Initial input position reference
    Eigen::VectorXd u_dot_0 = Eigen::VectorXd::Zero(3); // Initial input velocity reference (derivative of input position reference)
    Eigen::VectorXd u_ddot_0 = Eigen::VectorXd::Zero(3); // Initial input acceleration reference (derivative of input velocity reference)
    ConstraintConfig constraintConfig; // Configuration for enabling or disabling various constraints in the optimization problem
};
/**
 * @class Drone
 * @brief The Drone class is a subclass of AMSolver and is used to solve the drone trajectory optimization problem.
 */
class Drone : public AMSolver<DroneResult, DroneSolveArgs>{
public:
    /**
     * @struct MPCWeights
     * @brief Weights for cost terms in the optimization problem.
     */
    struct MPCWeights {
        double waypoints_pos = 7000; // waypoint position tracking
        double waypoints_vel = 1000; // waypoint velocity tracking
        double waypoints_acc = 100; // waypoint acceleration tracking
        double smoothness = 100; // smoothness of the trajectory
        double input_smoothness = 1000; // smoothness of the input
        double input_continuity = 100; // continuity of the input (see thesis document)

        MPCWeights() {}
        MPCWeights(double waypoints_pos, double waypoints_vel, double waypoints_acc, double smoothness, double input_smoothness, 
            double input_continuity) 
        : waypoints_pos(waypoints_pos), waypoints_vel(waypoints_vel), waypoints_acc(waypoints_acc), smoothness(smoothness), 
        input_smoothness(input_smoothness), input_continuity(input_continuity) {}
    };

    /**
     * @struct MPCConfig
     * @brief Parameters that are used to construct the MPC problem, which gets translated to an optimization problem.
     */
    struct MPCConfig {
        int K = 25; // number of time steps in the horizon
        int n = 10; // number of spline coefficients
        double mpc_freq = 8.0; // frequency of the MPC solver (Hz), used to calculate the time step
        double bf_gamma = 1.0; // barrier function gamma parameter (for collision constraints only)
        double waypoints_pos_tol = 1e-2; // tolerance for satisfaction of waypoint position constraint
        double waypoints_vel_tol = 1e-2; // tolerance for satisfaction of waypoint velocity constraint
        double waypoints_acc_tol = 1e-2; // tolerance for satisfaction of waypoint acceleration constraint
        double input_continuity_tol = 1e-2; // tolerance for satisfaction of input continuity constraint
        double pos_tol = 1e-2; // tolerance for satisfaction of position constraint
        double vel_tol = 1e-2; // tolerance for satisfaction of velocity constraint
        double acc_tol = 1e-2; // tolerance for satisfaction of acceleration constraint
        double collision_tol = 1e-2; // tolerance for satisfaction of collision constraint

        MPCConfig() {}
        MPCConfig(int K, int n, double mpc_freq, double bf_gamma, double waypoints_pos_tol,
                  double waypoints_vel_tol, double waypoints_acc_tol, double input_continuity_tol,
                  double pos_tol, double vel_tol, double acc_tol, double collision_tol)
                  : K(K), n(n), mpc_freq(mpc_freq), bf_gamma(bf_gamma),
                    waypoints_pos_tol(waypoints_pos_tol), waypoints_vel_tol(waypoints_vel_tol),
                    waypoints_acc_tol(waypoints_acc_tol), input_continuity_tol(input_continuity_tol),
                    pos_tol(pos_tol), vel_tol(vel_tol), acc_tol(acc_tol), collision_tol(collision_tol) {}
    };

    /**
     * @struct PhysicalLimits
     * @brief Physical limits for the drone, including position bounds and velocity/acceleration limits.
     */
    struct PhysicalLimits {
        Eigen::VectorXd p_min = Eigen::VectorXd::Constant(3,-10); // minimum position bounds
        Eigen::VectorXd p_max = Eigen::VectorXd::Constant(3,10); // maximum position bounds
        double v_bar = 1.73; // maximum velocity
        double a_bar = 0.75 * 9.81; // maximum acceleration (0.75g)
        double x_collision_envelope = 0.25; // collision envelope width in x direction
        double y_collision_envelope = 0.25; // collision envelope width in y direction
        double z_collision_envelope = 2.0 /3.0; // collision envelope width in z direction (larger due to downwash effect)

        PhysicalLimits() {}
        PhysicalLimits(const Eigen::VectorXd& p_min, const Eigen::VectorXd& p_max, double v_bar, double a_bar, double x_collision_envelope, double y_collision_envelope, double z_collision_envelope) 
        : p_min(p_min), p_max(p_max), v_bar(v_bar), a_bar(a_bar), x_collision_envelope(x_collision_envelope), y_collision_envelope(x_collision_envelope), z_collision_envelope(x_collision_envelope) {}
    };

    /**
     * @struct SparseDynamics
     * @brief Sparse matrices representing the dynamics of the drone.
     * The state is represented as [position, velocity] and the input is [position reference, velocity reference].
     * The state dynamics are given by the equation x[k+1] = Ax[k] + Bu[k], where x is the state and u is the input.
     * The derivative of the state [velocity, acceleration] is given by x'[k] = A'x[k] + B'u[k], where x' is the derivative of the state and u is the input.
     * See the thesis document for more information on the dynamics matrices and derivation of the derivative matrices.
     */
    struct SparseDynamics {
        Eigen::SparseMatrix<double> A, B, A_prime, B_prime;

        SparseDynamics() {}
        SparseDynamics(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, 
                const Eigen::SparseMatrix<double>& A_prime, const Eigen::SparseMatrix<double>& B_prime) 
        : A(A), B(B), A_prime(A_prime), B_prime(B_prime) {}
    };

    /**
     * Drone constructor.
     * Initializes a new instance of the Drone class.
     * 
     * @param solverConfig Configuration for the AMSolver.
     * @param waypoints Matrix of waypoints the drone should follow.
     * @param mpcConfig Configuration for the MPC optimization problem.
     * @param weights Weights for the MPC optimization problem.
     * @param limits Physical limits for the drone.
     * @param dynamics Dynamics matrices for the drone.
     */
    Drone(AMSolverConfig solverConfig,
            Eigen::MatrixXd waypoints,
            MPCConfig mpcConfig,
            MPCWeights weights,
            PhysicalLimits limits,
            SparseDynamics dynamics);
    
    // Getters
    Eigen::SparseMatrix<double> getCollisionEnvelope();
    int getK();

protected:
    /**
     * @struct SelectionMatrices
     * @brief Contains selection matrices used to extract specific components (position, velocity, and acceleration)
     * from the full state trajectory. These matrices are used for defining constraints and cost function terms
     * as functions of the overall state trajectory.
     */
    struct SelectionMatrices {
        Eigen::SparseMatrix<double> M_p, M_v, M_a; // pos,vel,acc

        /**
         * Constructor for SelectionMatrices.
         * Initializes the selection matrices based on the given horizon length (K).
         * 
         * @param K The number of time steps in the horizon.
         */
        SelectionMatrices(int K) {
            // Intermediate matrices used in building selection matrices
            Eigen::SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
            Eigen::SparseMatrix<double> eyeKplus1 = utils::getSparseIdentity(K+1);
            Eigen::SparseMatrix<double> zeroMat(3, 3);
            zeroMat.setZero();

            M_p = utils::kroneckerProduct(eyeKplus1, utils::horzcat(eye3, zeroMat));
            M_v = utils::kroneckerProduct(eyeKplus1, utils::horzcat(zeroMat, eye3));
            M_a = utils::kroneckerProduct(eyeKplus1, utils::horzcat(zeroMat, eye3));
        }
    };

    // The following members are sparse matrices used in the optimization problem. These include
    // matrices for weighting the cost function, selection matrices for extracting parts of the state,
    // and matrices representing the dynamics of the system.
    // These are member variables so that we can precompute them once and reuse them for each optimization solve step,
    // rather than recomputing them each time which is very slow

    Eigen::SparseMatrix<double> W, W_dot, W_ddot, W_input; // Bernstein matrices and derivates for parameterizing the input as spline
    Eigen::SparseMatrix<double> S_x, S_u, S_x_prime, S_u_prime; // State and input matrices for the full horizon (state condensing to write the trajectory as a function of the initial state and the input)

    Eigen::SparseMatrix<double> M_p_S_x, M_v_S_x, M_a_S_x_prime; // Precomputed matrix terms
    Eigen::SparseMatrix<double> G_u, G_u_T, G_u_T_G_u; 
    Eigen::SparseMatrix<double> G_p;
    Eigen::SparseMatrix<double> S_u_W_input;
    Eigen::SparseMatrix<double> M_p_S_u_W_input, M_v_S_u_W_input, M_a_S_u_prime_W_input;
    Eigen::SparseMatrix<double> linearCostSmoothnessConstTerm;

    // The following members are the configuration and parameters for the drone trajectory optimization problem
    MPCConfig mpcConfig;
    MPCWeights weights;
    PhysicalLimits limits;
    SparseDynamics dynamics;
    SelectionMatrices selectionMats;
    Eigen::MatrixXd waypoints;
    Eigen::SparseMatrix<double> collision_envelope; // this drone's collision envelope - NOT the other obstacles' collision envelopes

    // Protected methods
    /**
     * @brief Performs preprocessing steps before solving the optimization problem, override of AMSolver method.
     * This method sets up the optimization problem by configuring constraints and cost functions based on the current state,
     * input, and dynamic parameters of the drone.
     * 
     * @param args The arguments required for solving the drone trajectory optimization problem.
     */
    void preSolve(const DroneSolveArgs& args) override;

    /**
     * @brief Processes the solution of the optimization problem to generate a DroneResult object.
     * This method extracts the optimized trajectories and other relevant information from the solution
     * of the optimization problem and organizes them into a DroneResult object for easy access.
     * 
     * @param zeta The solution vector of the optimization problem.
     * @param args The arguments used for solving the drone trajectory optimization problem.
     * @return A DroneResult object containing the optimized trajectories and other results.
     */
    DroneResult postSolve(const Eigen::VectorXd& zeta, const DroneSolveArgs& args) override;

    /**
     * @brief Extracts waypoints within the current optimization horizon.
     * This method rounds the waypoint times to the nearest time step according to the frequency of the MPC solver,
     * and extracts the waypoints that fall within the current optimization horizon.
     * 
     * @param t The current time.
     * @return A matrix containing the waypoints within the current optimization horizon where each row is a waypoint,
     * with a waypoint given by [t, x, y, z, vx, vy, vz, ax, ay. az] (v = velocity, a = acceleration).
     */
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> extractWaypointsInCurrentHorizon(double t);

    /**
     * @brief Initializes the Bernstein matrices and their derivatives.
     * The input is parameterized as a spline using Bernstein polynomials, and these matrices are used to
     * represent the input and its derivatives as functions of the spline coefficients. The spline coefficients
     * are the optimization variables that are solved for in the optimization problem.
     * 
     * @param mpcConfig The configuration for the MPC optimization problem.
     * @return A tuple containing the Bernstein polynomial matrices and their first and second derivatives.
     */
    std::tuple<Eigen::SparseMatrix<double>,Eigen::SparseMatrix<double>,Eigen::SparseMatrix<double>,Eigen::SparseMatrix<double>> initBernsteinMatrices(const MPCConfig& mpcConfig);

    /**
     * @brief Initializes the full horizon dynamics matrices based on the drone's dynamics (state condensing),
     * which allows the drone's trajectory to be written as a function of the initial state and the input.
     * See thesis document for derivation.
     * 
     * @param dynamics The dynamics matrices of the drone.
     * @return A tuple containing the state and input selection matrices and their derivatives.
     */
    std::tuple<Eigen::SparseMatrix<double>,Eigen::SparseMatrix<double>,Eigen::SparseMatrix<double>,Eigen::SparseMatrix<double>> initFullHorizonDynamicsMatrices(const SparseDynamics& dynamics);
};

} // namespace amswarm

#endif
