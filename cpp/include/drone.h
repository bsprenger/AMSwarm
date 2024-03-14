#ifndef DRONE_H
#define DRONE_H

#include <amsolver.h>
#include <utils.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>


using namespace Eigen;

struct DroneResult {
    MatrixXd position_trajectory;
    VectorXd position_trajectory_vector;
    MatrixXd state_trajectory;
    VectorXd state_trajectory_vector;
    MatrixXd input_position_trajectory;
    VectorXd input_position_trajectory_vector;
    MatrixXd input_velocity_trajectory;
    VectorXd input_velocity_trajectory_vector;
    MatrixXd input_acceleration_trajectory;
    VectorXd input_acceleration_trajectory_vector;
    VectorXd spline_coeffs;

    void advanceForNextSolveStep();
    static DroneResult generateInitialDroneResult(const VectorXd& initial_position, int K);
};

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
};

struct DroneSolveArgs {
    double current_time = 0.0;
    int num_obstacles = 0;
    std::vector<SparseMatrix<double>> obstacle_envelopes = {};
    std::vector<VectorXd> obstacle_positions = {};
    VectorXd x_0 = VectorXd::Zero(6);
    VectorXd u_0 = VectorXd::Zero(3);
    VectorXd u_dot_0 = VectorXd::Zero(3);
    VectorXd u_ddot_0 = VectorXd::Zero(3);
    ConstraintConfig constraintConfig;
};

class Drone : public AMSolver<DroneResult, DroneSolveArgs>{
public:
    struct MPCWeights {
        double waypoints_pos = 7000;
        double waypoints_vel = 1000;
        double waypoints_acc = 100;
        double smoothness = 100;
        double input_smoothness = 1000;
        double input_continuity = 100;

        MPCWeights() {}
        MPCWeights(double waypoints_pos, double waypoints_vel, double waypoints_acc, double smoothness, double input_smoothness, 
            double input_continuity) 
        : waypoints_pos(waypoints_pos), waypoints_vel(waypoints_vel), waypoints_acc(waypoints_acc), smoothness(smoothness), 
        input_smoothness(input_smoothness), input_continuity(input_continuity) {}
    };

    struct MPCConfig {
        int K = 25;
        int n = 10;
        double mpc_freq = 8.0;
        double bf_gamma = 1.0;
        double waypoints_pos_tol = 1e-2;
        double waypoints_vel_tol = 1e-2;
        double waypoints_acc_tol = 1e-2;
        double input_continuity_tol = 1e-2;
        double pos_tol = 1e-2;
        double vel_tol = 1e-2;
        double acc_tol = 1e-2;
        double collision_tol = 1e-2;

        MPCConfig() {}
        MPCConfig(int K, int n, double mpc_freq, double bf_gamma, double waypoints_pos_tol,
                  double waypoints_vel_tol, double waypoints_acc_tol, double input_continuity_tol,
                  double pos_tol, double vel_tol, double acc_tol, double collision_tol)
                  : K(K), n(n), mpc_freq(mpc_freq), bf_gamma(bf_gamma),
                    waypoints_pos_tol(waypoints_pos_tol), waypoints_vel_tol(waypoints_vel_tol),
                    waypoints_acc_tol(waypoints_acc_tol), input_continuity_tol(input_continuity_tol),
                    pos_tol(pos_tol), vel_tol(vel_tol), acc_tol(acc_tol), collision_tol(collision_tol) {}
    };

    struct PhysicalLimits {
        VectorXd p_min = VectorXd::Constant(3,-10);
        VectorXd p_max = VectorXd::Constant(3,10);
        double v_bar = 1.73;
        double a_bar = 0.75 * 9.81;

        PhysicalLimits() {}
        PhysicalLimits(const Eigen::VectorXd& p_min, const Eigen::VectorXd& p_max, double v_bar, double a_bar) 
        : p_min(p_min), p_max(p_max), v_bar(v_bar), a_bar(a_bar) {}
    };

    struct SparseDynamics {
        SparseMatrix<double> A, B, A_prime, B_prime;

        SparseDynamics() {}
        SparseDynamics(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, 
                const Eigen::SparseMatrix<double>& A_prime, const Eigen::SparseMatrix<double>& B_prime) 
        : A(A), B(B), A_prime(A_prime), B_prime(B_prime) {}
    };

    // Constructors
    Drone(AMSolverConfig solverConfig,
            MatrixXd waypoints,
            MPCConfig mpcConfig,
            MPCWeights weights,
            PhysicalLimits limits,
            SparseDynamics dynamics);
    
    // Getters TODO check if these are necessary
    SparseMatrix<double> getCollisionEnvelope();
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
    MPCConfig mpcConfig;
    MPCWeights weights;
    PhysicalLimits limits;
    SparseDynamics dynamics;
    SelectionMatrices selectionMats;
    MatrixXd waypoints;
    SparseMatrix<double> collision_envelope; // this drone's collision envelope - NOT the other obstacles' collision envelopes

    // Protected methods
    void preSolve(const DroneSolveArgs& args) override;
    DroneResult postSolve(const VectorXd& zeta, const DroneSolveArgs& args) override;
    Matrix<double, Dynamic, Dynamic, RowMajor> extractWaypointsInCurrentHorizon(double t);
    std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> initBernsteinMatrices(const MPCConfig& mpcConfig);
    std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> initFullHorizonDynamicsMatrices(const SparseDynamics& dynamics);
};

#endif
