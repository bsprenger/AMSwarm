#ifndef DRONE_H
#define DRONE_H

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>


class Drone {
    public:
        // Public struct definitions

        class DroneResult {
            public:
                Eigen::VectorXd position_state_time_stamps; // time stamps for both position and state
                Eigen::VectorXd control_input_time_stamps;

                Eigen::MatrixXd position_trajectory;
                Eigen::MatrixXd state_trajectory;
                Eigen::MatrixXd control_input_trajectory;

                Eigen::VectorXd position_trajectory_vector;
                Eigen::VectorXd state_trajectory_vector;
                Eigen::VectorXd control_input_trajectory_vector;
        };

        // Constructors
        Drone(std::string& params_filepath, // necessary input
                Eigen::MatrixXd waypoints, // necessary input
                Eigen::VectorXd initial_pos = Eigen::VectorXd::Zero(3), // optional inputs - default values are set
                bool hard_waypoint_constraints = true,
                bool acceleration_constraints = true,
                int K = 25,
                int n = 10,
                float delta_t = 1.0/6.0, // FIX THIS cast to float always
                Eigen::VectorXd p_min = Eigen::VectorXd::Constant(3,-10),
                Eigen::VectorXd p_max = Eigen::VectorXd::Constant(3,10),
                float w_g_p = 7000,
                float w_g_v = 1000,
                float w_s = 100,
                float v_bar = 1.73,
                float f_bar = 1.5*9.8);

        // Public methods
        DroneResult solve(const double, const Eigen::VectorXd, const int, const std::vector<Eigen::SparseMatrix<double>>, const Eigen::VectorXd);
        
        // Getters
        Eigen::VectorXd getInitialPosition();
        Eigen::SparseMatrix<double> getCollisionEnvelope();
        Eigen::MatrixXd getWaypoints();
        float getDeltaT();
        int getK();

        // Setters
        // To do


    private:
        // Private struct definitions 
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

            Residuals(int j, int K, int num_penalized_steps) {
                eq = Eigen::VectorXd::Ones((2 + j) * 3 * K); // TODO something more intelligent then setting these to 1 -> they should be bigger than threshold
                pos = Eigen::VectorXd::Ones(6 * K);
                waypoints = Eigen::VectorXd::Ones(6 * num_penalized_steps);
                accel = Eigen::VectorXd::Ones(6 * num_penalized_steps);
            }
        };

        struct LagrangeMultipliers {
            Eigen::VectorXd eq; // equality constraint residuals
            Eigen::VectorXd pos; // position constraint residuals
            Eigen::VectorXd waypoints; // waypoint constraint residuals
            Eigen::VectorXd accel; // acceleration constraint residuals --> to do change this

            LagrangeMultipliers(int j, int K, int num_penalized_steps) {
                eq = Eigen::VectorXd::Zero((2 + j) * 3 * K);
                pos = Eigen::VectorXd::Zero(6 * K);
                waypoints = Eigen::VectorXd::Zero(6 * num_penalized_steps);
                accel = Eigen::VectorXd::Zero(6 * num_penalized_steps);
            }
        };

        struct CostMatrices {
            Eigen::SparseMatrix<double> Q, q; // why is q a matrix? --> clarify this later
            Eigen::SparseMatrix<double> A_check_const_terms;
            Eigen::SparseMatrix<double> A_check;
            Eigen::SparseVector<double> b_check;
        };

        
        // Private variables
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
        bool hard_waypoint_constraints;
        bool acceleration_constraints;
        Eigen::MatrixXd waypoints;
        Eigen::VectorXd initial_pos;
        Eigen::SparseMatrix<double> collision_envelope; // this drone's collision envelope - NOT the other obstacles' collision envelopes

        // Private methods
        ConstSelectionMatrices constSelectionMatrices;
        void initConstSelectionMatrices(); // to do remove this for a default constructor


        Eigen::MatrixXd extractWaypointsInCurrentHorizon(const double, const Eigen::MatrixXd&);
        void generateBernsteinMatrices();
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> loadDynamicsMatricesFromFile(const std::string&);
        std::tuple<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> loadSparseDynamicsMatricesFromFile(const std::string&);
        void generateFullHorizonDynamicsMatrices(std::string&);

        void initOptimizationParams(Eigen::MatrixXd& extracted_waypoints,
                                    Eigen::VectorXd& penalized_steps,
                                    int j,
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

        DroneResult computeDroneResult(double current_time, Eigen::VectorXd& zeta_1,Eigen::VectorXd x_0);

        void printUnsatisfiedResiduals(const Residuals& residuals, double threshold);
};

#endif
