#ifndef DRONE_H
#define DRONE_H

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <vector>

class Drone {
    public:
        Drone(std::string& params_filepath, Eigen::MatrixXd waypoints, // necessary inputs
                Eigen::VectorXd initial_pos = Eigen::VectorXd::Zero(3), // optional inputs - default values are set
                int K = 25, int n = 10, float delta_t = 1/6,
                Eigen::VectorXd p_min = Eigen::VectorXd::Constant(3,-10),
                Eigen::VectorXd p_max = Eigen::VectorXd::Constant(3,10),
                float w_g_p = 7000, float w_g_v = 1000, float w_s = 100,
                float v_bar = 1.73, float f_bar = 1.5*9.8);

        void solve(const double, const Eigen::VectorXd, const int, const std::vector<Eigen::SparseMatrix<double>>, const Eigen::VectorXd);

        int K;
        int n;
        float delta_t;
        float t_f;
        Eigen::VectorXd p_min;
        Eigen::VectorXd p_max;
        float w_g_p;
        float w_g_v;
        float w_s;
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

        bool hard_waypoint_constraints = true;

    private:
        Eigen::MatrixXd extractWaypointsInCurrentHorizon(const double, const Eigen::MatrixXd&);
        void generateBernsteinMatrices();
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> loadDynamicsMatricesFromFile(const std::string&);
        std::tuple<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> loadSparseDynamicsMatricesFromFile(const std::string&);
        void generateFullHorizonDynamicsMatrices(std::string&);

        void initOptimizationParams(int j,
                                    Eigen::VectorXd x_0,
                                    Eigen::VectorXd xi,
                                    double current_time,
                                    Eigen::SparseMatrix<double>& M_p,
                                    Eigen::SparseMatrix<double>& M_v,
                                    Eigen::SparseMatrix<double>& M_a,
                                    Eigen::SparseMatrix<double>& M_x,
                                    Eigen::SparseMatrix<double>& M_y,
                                    Eigen::SparseMatrix<double>& M_z,
                                    std::vector<Eigen::SparseMatrix<double>>& thetas,
                                    Eigen::VectorXd& alpha,
                                    Eigen::VectorXd& beta,
                                    Eigen::VectorXd& d,
                                    Eigen::VectorXd& zeta_1,
                                    Eigen::SparseMatrix<double>& G_waypoints,
                                    Eigen::SparseMatrix<double>& G_eq,
                                    Eigen::SparseMatrix<double>& G_pos,
                                    Eigen::VectorXd& c_waypoints,
                                    Eigen::VectorXd& c_eq,
                                    Eigen::VectorXd& h_waypoints,
                                    Eigen::VectorXd& h_pos,
                                    Eigen::VectorXd& h_eq, Eigen::VectorXd& s,
                                    Eigen::VectorXd& res_eq,
                                    Eigen::VectorXd& res_pos,
                                    Eigen::VectorXd& res_waypoints,
                                    Eigen::VectorXd& lambda_eq,
                                    Eigen::VectorXd& lambda_pos,
                                    Eigen::VectorXd& lambda_waypoints,
                                    Eigen::SparseMatrix<double>& Q,
                                    Eigen::SparseMatrix<double>& q, 
                                    Eigen::SparseMatrix<double>& A_check_const_terms);

        void initSelectionMatrices(int, Eigen::VectorXd&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&);
        void initOptimizationVariables(int, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&);
        void initConstConstraintMatrices(int, Eigen::VectorXd, Eigen::VectorXd, Eigen::SparseMatrix<double>&, Eigen::MatrixXd&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&,Eigen::VectorXd&,Eigen::VectorXd&,Eigen::VectorXd&,Eigen::VectorXd&);
        void initCostMatrices(Eigen::VectorXd&, Eigen::SparseMatrix<double>&, Eigen::VectorXd, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&);
        void initResiduals(int j, int num_penalized_steps, Eigen::VectorXd& res_eq, Eigen::VectorXd& res_pos, Eigen::VectorXd& res_waypoints);
        void initLagrangeMultipliers(int j, int num_penalized_steps, Eigen::VectorXd& lambda_eq, Eigen::VectorXd& lambda_pos, Eigen::VectorXd& lambda_waypoints);

        void computeX_g(Eigen::MatrixXd& extracted_waypoints, Eigen::VectorXd& penalized_steps, Eigen::SparseMatrix<double>& X_g);

        void computeZeta1(double rho,
                        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>& solver,
                        Eigen::SparseMatrix<double>& A_check,
                        Eigen::SparseVector<double>& b_check,
                        Eigen::SparseMatrix<double>& A_check_const_terms,
                        Eigen::SparseMatrix<double>& Q,
                        Eigen::SparseMatrix<double>& q,
                        Eigen::SparseMatrix<double>& G_waypoints,
                        Eigen::SparseMatrix<double>& G_eq,
                        Eigen::SparseMatrix<double>& G_pos,
                        Eigen::VectorXd& c_waypoints,
                        Eigen::VectorXd& c_eq, 
                        Eigen::VectorXd& h_waypoints,
                        Eigen::VectorXd& h_pos,
                        Eigen::VectorXd& h_eq,
                        Eigen::VectorXd& s,
                        Eigen::VectorXd& lambda_eq,
                        Eigen::VectorXd& lambda_pos,
                        Eigen::VectorXd& lambda_waypoints,
                        Eigen::VectorXd& zeta_1);

        void compute_h_eq(int, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&,Eigen::VectorXd&);
        void compute_d(int, int, double, Eigen::SparseMatrix<double>&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&);
        void computeAlphaBeta(double, Eigen::SparseMatrix<double>&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::VectorXd&, Eigen::VectorXd&);
        void computeResiduals(Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::SparseMatrix<double>&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&);
        void updateLagrangeMultipliers(double rho, Eigen::VectorXd& res_eq, Eigen::VectorXd& res_pos, Eigen::VectorXd& res_waypoints, Eigen::VectorXd& lambda_eq, Eigen::VectorXd& lambda_pos, Eigen::VectorXd& lambda_waypoints);

        void computeInputOverHorizon(Eigen::VectorXd& zeta_1);
        void computeStatesOverHorizon(const Eigen::VectorXd x_0);
        void computePositionOverHorizon(Eigen::SparseMatrix<double>& M_p);
};

#endif
