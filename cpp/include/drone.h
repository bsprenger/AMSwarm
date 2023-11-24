#ifndef DRONE_H
#define DRONE_H

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <vector>

class Drone {
    public:
        Drone(int K, int n, float delta_t, Eigen::VectorXd p_min, Eigen::VectorXd p_max, float w_g_p, float w_g_v, float w_s, int kappa, float v_bar, float f_bar, Eigen::VectorXd initial_pos, Eigen::MatrixXd waypoints, std::string& params_filepath);

        void solve(const double, const Eigen::VectorXd, const int, const std::vector<Eigen::SparseMatrix<double>>, const Eigen::VectorXd);
        Eigen::MatrixXd extractWaypointsInCurrentHorizon(const double, const Eigen::MatrixXd&);
        void generateAndAssignBernsteinMatrices();
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> loadDynamicsMatricesFromFile(const std::string&);
        std::tuple<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> loadSparseDynamicsMatricesFromFile(const std::string&);
        void generateFullHorizonDynamicsMatrices(std::string&);

        int K;
        int n;
        float delta_t;
        float t_f;
        Eigen::VectorXd p_min;
        Eigen::VectorXd p_max;
        float w_g_p;
        float w_g_v;
        float w_s;
        int kappa;
        double v_bar;
        double f_bar;

        Eigen::SparseMatrix<double> W, W_dot;
        Eigen::SparseMatrix<double> S_x, S_u, S_x_prime, S_u_prime;

        Eigen::MatrixXd waypoints;

        Eigen::VectorXd input_traj_vector;
        Eigen::VectorXd state_traj_vector;
        Eigen::VectorXd pos_traj_vector;
        Eigen::MatrixXd input_traj_matrix; // each column is a time step. each row contains an x, y, or z input position
        Eigen::MatrixXd state_traj_matrix;
        Eigen::MatrixXd pos_traj_matrix;

        Eigen::SparseMatrix<double> collision_envelope; // this drone's collision envelope - NOT the other obstacles' collision envelopes

        bool hard_waypoint_constraints = false;
};

#endif