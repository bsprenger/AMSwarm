#ifndef DRONE_H
#define DRONE_H

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>


class Drone {
    public:
        struct OptimizationResult {
            Eigen::VectorXd input_traj_vector;
            Eigen::VectorXd state_traj_vector;
            Eigen::VectorXd pos_traj_vector;
            Eigen::MatrixXd input_traj_matrix; // each column is a time step. each row contains an x, y, or z input position
            Eigen::MatrixXd state_traj_matrix;
            Eigen::MatrixXd pos_traj_matrix;
        };

        Drone(std::string& params_filepath, Eigen::MatrixXd waypoints, // necessary inputs
                Eigen::VectorXd initial_pos = Eigen::VectorXd::Zero(3), // optional inputs - default values are set
                int K = 25, int n = 10, float delta_t = 1.0/6.0, // FIX THIS cast to float always
                Eigen::VectorXd p_min = Eigen::VectorXd::Constant(3,-10),
                Eigen::VectorXd p_max = Eigen::VectorXd::Constant(3,10),
                float w_g_p = 7000, float w_g_v = 1000, float w_s = 100,
                float v_bar = 1.73, float f_bar = 1.5*9.8);

        OptimizationResult solve(const double, const Eigen::VectorXd, const int, const std::vector<Eigen::SparseMatrix<double>>, const Eigen::VectorXd);

        // to do: make private or protected
        Eigen::SparseMatrix<double> collision_envelope; // this drone's collision envelope - NOT the other obstacles' collision envelopes
        
        // Getters TO DO
        Eigen::VectorXd getInitialPosition();

    private:
        struct ConstSelectionMatrices {
            Eigen::SparseMatrix<double> M_p, M_v, M_a; // maybe rename to p,v,a
        };

        struct VariableSelectionMatrices {
            Eigen::SparseMatrix<double> M_x, M_y, M_z, M_waypoints_penalized; // maybe rename to x,y,z,timestep?
        };

        struct Constraints {
            Eigen::SparseMatrix<double> G_eq, G_pos, G_waypoints, G_accel;
            Eigen::VectorXd h_eq, h_pos, h_waypoints, h_accel;
            Eigen::VectorXd c_eq, c_waypoints, c_accel;
        };

        struct Residuals {
            Eigen::VectorXd eq; // equality constraint residuals
            Eigen::VectorXd pos; // position constraint residuals
            Eigen::VectorXd waypoints; // waypoint constraint residuals
            Eigen::VectorXd accel; // acceleration constraint residuals --> to do change this
        };

        struct LagrangeMultipliers {
            Eigen::VectorXd eq; // equality constraint residuals
            Eigen::VectorXd pos; // position constraint residuals
            Eigen::VectorXd waypoints; // waypoint constraint residuals
            Eigen::VectorXd accel; // acceleration constraint residuals --> to do change this
        };

        struct CostMatrices {
            Eigen::SparseMatrix<double> Q, q; // why is q a matrix? --> clarify this later
            Eigen::SparseMatrix<double> A_check_const_terms;
            Eigen::SparseMatrix<double> A_check;
            Eigen::SparseVector<double> b_check;
        };

        
        Eigen::SparseMatrix<double> W, W_dot;
        Eigen::SparseMatrix<double> S_x, S_u, S_x_prime, S_u_prime;
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
        bool hard_waypoint_constraints = true;
        bool acceleration_constraints = true;
        Eigen::MatrixXd waypoints;
        Eigen::VectorXd initial_pos;

        ConstSelectionMatrices constSelectionMatrices;
        void initConstSelectionMatrices();


        Eigen::MatrixXd extractWaypointsInCurrentHorizon(const double, const Eigen::MatrixXd&);
        void generateBernsteinMatrices();
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> loadDynamicsMatricesFromFile(const std::string&);
        std::tuple<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> loadSparseDynamicsMatricesFromFile(const std::string&);
        void generateFullHorizonDynamicsMatrices(std::string&);

        void initOptimizationParams(int j,
                                    Eigen::VectorXd x_0,
                                    Eigen::VectorXd xi,
                                    double current_time,
                                    std::vector<Eigen::SparseMatrix<double>>& thetas,
                                    Eigen::VectorXd& alpha,
                                    Eigen::VectorXd& beta,
                                    Eigen::VectorXd& d,
                                    Eigen::VectorXd& zeta_1,
                                    Eigen::VectorXd& s,
                                    VariableSelectionMatrices& variableSelectionMatrices,
                                    Residuals& residuals,
                                    LagrangeMultipliers& lambda,
                                    Constraints& constraints,
                                    CostMatrices& costMatrices);

        void initVariableSelectionMatrices(int j, Eigen::VectorXd& penalized_steps,
                            VariableSelectionMatrices& variableSelectionMatrices);
                            
        void initOptimizationVariables(int, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&);

        void initConstConstraintMatrices(int j, Eigen::VectorXd x_0,
                                Eigen::VectorXd xi,
                                Eigen::SparseMatrix<double>& S_theta,
                                Eigen::MatrixXd& extracted_waypoints,
                                Eigen::SparseMatrix<double>& M_waypoints_penalized,
                                Constraints& constraints);

        void initCostMatrices(Eigen::VectorXd& penalized_steps,
                            Eigen::VectorXd x_0,
                            Eigen::SparseMatrix<double>& X_g,
                            CostMatrices& costMatrices);

        void initResiduals(int j, int num_penalized_steps, Residuals& residuals);
        void initLagrangeMultipliers(int j, int num_penalized_steps,
                                    LagrangeMultipliers& lambda);

        void computeX_g(Eigen::MatrixXd& extracted_waypoints, Eigen::VectorXd& penalized_steps, Eigen::SparseMatrix<double>& X_g);

        void computeZeta1(int iters, double rho,
                        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
                        CostMatrices& costMatrices,
                        Constraints& constraints,
                        Eigen::VectorXd& s,
                        LagrangeMultipliers& lambda,
                        Eigen::VectorXd& zeta_1);

        void compute_h_eq(int, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&,Eigen::VectorXd&);
        void compute_d(int K, int j, double rho,
                    Constraints& constraints, Eigen::VectorXd& zeta_1,
                    LagrangeMultipliers& lambda,
                    Eigen::VectorXd& alpha, Eigen::VectorXd& beta,
                    Eigen::VectorXd& d);
        void computeAlphaBeta(double rho, Constraints& constraints,
                            Eigen::VectorXd& zeta_1,
                            LagrangeMultipliers& lambda, 
                            Eigen::VectorXd& alpha, Eigen::VectorXd& beta,
                            VariableSelectionMatrices& variableSelectionMatrices);
                            
        void computeResiduals(Constraints& constraints,
                            Eigen::VectorXd& zeta_1, Eigen::VectorXd& s,
                            Residuals& residuals);

        void updateLagrangeMultipliers(double rho, Residuals& residuals, LagrangeMultipliers& lambda);

        OptimizationResult computeOptimizationResult(Eigen::VectorXd& zeta_1,Eigen::VectorXd x_0);
};

#endif
