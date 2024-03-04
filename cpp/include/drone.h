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
    MatrixXd position_trajectory;
    MatrixXd state_trajectory;
    MatrixXd control_input_trajectory;
    VectorXd position_trajectory_vector;
    VectorXd state_trajectory_vector;
    VectorXd control_input_trajectory_vector;
    VectorXd spline_coeffs;
};

struct DroneSolveArgs {
    double current_time = 0.0;
    bool waypoint_position_constraints = true;
    bool waypoint_velocity_constraints = true;
    bool waypoint_acceleration_constraints = false;
    bool input_continuity_constraints = true;
    int max_iters = 1000;
    double rho_init = 1.3;
    int num_obstacles = 0;
    std::vector<SparseMatrix<double>> obstacle_envelopes = {};
    std::vector<VectorXd> obstacle_positions = {};
    VectorXd x_0 = VectorXd::Zero(6);
};

struct MPCWeights {
    double waypoints_pos = 7000;
    double waypoints_vel = 1000;
    double waypoints_acc = 100;
    double smoothness = 100;
    double input_smoothness = 1000;
    double input_continuity = 100;

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

class Drone : public AMSolver<DroneResult, DroneSolveArgs>{
    public:
        // Constructors
        Drone(MatrixXd waypoints,
                MPCConfig config,
                MPCWeights weights,
                PhysicalLimits limits,
                SparseDynamics dynamics,
                VectorXd initial_pos);

        // Public methods
        
        
        // Getters
        VectorXd getInitialPosition();
        SparseMatrix<double> getCollisionEnvelope();
        MatrixXd getWaypoints();
        float getDeltaT();
        int getK();

    protected:
        // Protected struct definitions 
        struct SelectionMatrices {
            SparseMatrix<double> M_p, M_v, M_a; // maybe rename to pos,vel,acc

            SelectionMatrices(int K) {
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

        // Protected variables
        SparseMatrix<double> W, W_dot, W_ddot, W_input;
        SparseMatrix<double> S_x, S_u, S_x_prime, S_u_prime;
        MPCConfig config;
        MPCWeights weights;
        PhysicalLimits limits;
        SparseDynamics dynamics;
        SelectionMatrices selectionMats;
        MatrixXd waypoints;
        VectorXd initial_pos;
        SparseMatrix<double> collision_envelope; // this drone's collision envelope - NOT the other obstacles' collision envelopes

        // Protected methods
        void preSolve(const DroneSolveArgs& args) override;
        DroneResult postSolve(const VectorXd& zeta, const DroneSolveArgs& args) override;
        MatrixXd extractWaypointsInCurrentHorizon(double t);
        std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> initBernsteinMatrices(const MPCConfig& config);
        std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> initFullHorizonDynamicsMatrices(const SparseDynamics& dynamics);
};

#endif
