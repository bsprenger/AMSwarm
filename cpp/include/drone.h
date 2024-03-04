#ifndef DRONE_H
#define DRONE_H

#include <utils.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>


using namespace Eigen;

struct DroneResult {
    VectorXd position_state_time_stamps; // time stamps for both position and state
    VectorXd control_input_time_stamps;
    MatrixXd position_trajectory;
    MatrixXd state_trajectory;
    MatrixXd control_input_trajectory;
    VectorXd position_trajectory_vector;
    VectorXd state_trajectory_vector;
    VectorXd control_input_trajectory_vector;
    VectorXd spline_coeffs;
};

struct SolveOptions {
    bool waypoint_position_constraints = true;
    bool waypoint_velocity_constraints = true;
    bool waypoint_acceleration_constraints = false;
    bool input_continuity_constraints = true;
    bool input_dot_continuity_constraints = true;
    bool input_ddot_continuity_constraints = true;
    int max_iters = 1000;
    double rho_init = 1.3;
    double eq_threshold = 0.01;
    double pos_threshold = 0.01;
    double waypoint_position_threshold = 0.01;
    double waypoint_velocity_threshold = 0.05;
    double waypoint_acceleration_threshold = 0.01;
    double input_continuity_threshold = 0.01;
    double input_dot_continuity_threshold = 0.01;
    double input_ddot_continuity_threshold = 0.01;
};

struct MPCWeights {
    double w_goal_pos = 7000;
    double w_goal_vel = 1000;
    double w_smoothness = 100;
    double w_input_smoothness = 1000;
    double w_input_continuity = 100;
    double w_input_dot_continuity = 100;
    double w_input_ddot_continuity = 100;

    MPCWeights() {}
    MPCWeights(double goal_pos, double goal_vel, double smoothness, double input_smoothness, 
        double input_continuity, double input_dot_continuity, double input_ddot_continuity) 
    : w_goal_pos(goal_pos), w_goal_vel(goal_vel), w_smoothness(smoothness), 
    w_input_smoothness(input_smoothness), w_input_continuity(input_continuity), 
    w_input_dot_continuity(input_dot_continuity), w_input_ddot_continuity(input_ddot_continuity) {}
};

struct MPCConfig {
    int K = 25;
    int n = 10;
    double delta_t = 1.0/8.0;
    double bf_gamma = 1.0;

    MPCConfig() {}
    MPCConfig(int K, int n, double delta_t, double bf_gamma) : K(K), n(n), delta_t(delta_t), bf_gamma(bf_gamma) {}
};

struct PhysicalLimits {
    VectorXd p_min = VectorXd::Constant(3,-10);
    VectorXd p_max = VectorXd::Constant(3,10);
    double v_bar = 1.73;
    double f_bar = 0.75 * 9.81;

    PhysicalLimits() {}
    PhysicalLimits(const Eigen::VectorXd& p_min, const Eigen::VectorXd& p_max, double v_bar, double f_bar) 
    : p_min(p_min), p_max(p_max), v_bar(v_bar), f_bar(f_bar) {}
};

struct SparseDynamics {
    SparseMatrix<double> A, B, A_prime, B_prime;

    SparseDynamics() {}
    SparseDynamics(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, 
            const Eigen::SparseMatrix<double>& A_prime, const Eigen::SparseMatrix<double>& B_prime) 
    : A(A), B(B), A_prime(A_prime), B_prime(B_prime) {}
};


class Drone : public AMSolver<DroneResult, DroneArgs>{
    public:
        // Constructors
        Drone(MatrixXd waypoints, // necessary input
                MPCConfig config,
                MPCWeights weights,
                PhysicalLimits limits,
                SparseDynamics dynamics,
                VectorXd initial_pos);

        // Public methods
        void preSolve(const DroneArgs& args) override;
        DroneResult postSolve(const VectorXd& zeta, const DroneArgs& args) override;
        
        // Getters
        VectorXd getInitialPosition();
        SparseMatrix<double> getCollisionEnvelope();
        MatrixXd getWaypoints();
        float getDeltaT();
        int getK();

    private:
        // Private struct definitions 
        struct ConstSelectionMatrices {
            SparseMatrix<double> M_p, M_v, M_a; // maybe rename to pos,vel,acc

            ConstSelectionMatrices(int K) {
                // Intermediate matrices used in building selection matrices
                SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
                SparseMatrix<double> eyeKplus1 = utils::getSparseIdentity(K+1);
                SparseMatrix<double> zeroMat(3, 3);
                zeroMat.setZero();

                M_p = utils::kroneckerProduct(eyeKplus1, utils::horzcat(eye3, zeroMat));
                M_v = utils::kroneckerProduct(eyeKplus1, utils::horzcat(zeroMat, eye3));
                M_a = utils::kroneckerProduct(eyeKplus1, utils::horzcat(zeroMat, eye3));
            }
        };

        struct VariableSelectionMatrices {
            SparseMatrix<double> M_x, M_y, M_z, M_waypoints_position, M_waypoints_velocity; // maybe rename to x,y,z,timestep?

            VariableSelectionMatrices(int K, int j, VectorXd& penalized_steps) {
                SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
                SparseMatrix<double> eye6 = utils::getSparseIdentity(6);
                SparseMatrix<double> eyeK = utils::getSparseIdentity(K);
                SparseMatrix<double> eyeKplus12j = utils::getSparseIdentity((2 + j) * (K+1));
                SparseMatrix<double> zeroMat(3, 3);
                zeroMat.setZero();
                SparseMatrix<double> x_step(1, 3);
                x_step.coeffRef(0, 0) = 1.0;
                SparseMatrix<double> y_step(1, 3);
                y_step.coeffRef(0, 1) = 1.0;
                SparseMatrix<double> z_step(1, 3);
                z_step.coeffRef(0, 2) = 1.0;

                M_x = utils::kroneckerProduct(eyeKplus12j, x_step);
                M_y = utils::kroneckerProduct(eyeKplus12j, y_step);
                M_z = utils::kroneckerProduct(eyeKplus12j, z_step);

                M_waypoints_position.resize(3 * penalized_steps.size(), 6 * (K+1));
                for (int i = 0; i < penalized_steps.size(); ++i) {
                    utils::replaceSparseBlock(M_waypoints_position, eye3, 3 * i, 6 * (penalized_steps(i))); // CHECK THIS
                }

                M_waypoints_velocity.resize(3 * penalized_steps.size(), 6 * (K+1));
                for (int i = 0; i < penalized_steps.size(); ++i) {
                    utils::replaceSparseBlock(M_waypoints_velocity, eye3, 3 * i, 6 * (penalized_steps(i)) + 3); // CHECK THIS
                }
            }
        };

        
        // Private variables
        SparseMatrix<double> W, W_dot, W_ddot, W_input;
        SparseMatrix<double> S_x, S_u, S_x_prime, S_u_prime;
        MPCConfig config;
        MPCWeights weights;
        PhysicalLimits limits;
        SparseDynamics dynamics;

        MatrixXd waypoints;
        VectorXd initial_pos;
        SparseMatrix<double> collision_envelope; // this drone's collision envelope - NOT the other obstacles' collision envelopes

        // Private methods
        ConstSelectionMatrices constSelectionMatrices;

        MatrixXd extractWaypointsInCurrentHorizon(const double t,
                                                        const MatrixXd& waypoints);
        std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> initBernsteinMatrices(const MPCConfig& config);
        std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> initFullHorizonDynamicsMatrices(const SparseDynamics& dynamics);

        DroneResult computeDroneResult(double current_time, VectorXd& zeta_1,VectorXd x_0);

        void printUnsatisfiedResiduals(const Residuals& residuals,
                                        SolveOptions& opt);

};

#endif
