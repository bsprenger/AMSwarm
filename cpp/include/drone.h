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

                bool is_successful;
        };

        // Constructors
        Drone(std::string& params_filepath, // necessary input
                Eigen::MatrixXd waypoints, // necessary input
                Eigen::VectorXd initial_pos = Eigen::VectorXd::Zero(3), // optional inputs - default values are set
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
        DroneResult solve(const double current_time,
                                const Eigen::VectorXd x_0,
                                Eigen::VectorXd& initial_guess_control_input_trajectory_vector,
                                const int j,
                                std::vector<Eigen::SparseMatrix<double>> thetas,
                                const Eigen::VectorXd xi,
                                bool waypoint_position_constraints,
                                bool waypoint_velocity_constraints,
                                bool waypoint_acceleration_constraints);
        
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
            Eigen::SparseMatrix<double> M_x, M_y, M_z, M_waypoints_position, M_waypoints_velocity; // maybe rename to x,y,z,timestep?
        };

        struct Constraints {
            Eigen::SparseMatrix<double> G_eq, G_pos, G_waypoints_pos,G_waypoints_vel, G_waypoints_accel;
            Eigen::VectorXd h_eq, h_pos, h_waypoints_pos, h_waypoints_vel, h_waypoints_accel;
            Eigen::VectorXd c_eq, c_waypoints_pos, c_waypoints_vel, c_waypoints_accel;
        };

        struct Residuals {
            Eigen::VectorXd eq; // equality constraint residuals
            Eigen::VectorXd pos; // position constraint residuals
            Eigen::VectorXd waypoints_pos; // waypoint constraint residuals
            Eigen::VectorXd waypoints_vel; // waypoint constraint residuals
            Eigen::VectorXd waypoints_accel; // acceleration constraint residuals --> to do change this
            Eigen::VectorXd u_0;

            Residuals(int j, int K, int num_penalized_steps) {
                eq = Eigen::VectorXd::Ones((2 + j) * 3 * K); // TODO something more intelligent then setting these to 1 -> they should be bigger than threshold
                pos = Eigen::VectorXd::Ones(6 * K);
                waypoints_pos = Eigen::VectorXd::Ones(3 * num_penalized_steps);
                waypoints_vel = Eigen::VectorXd::Ones(3 * num_penalized_steps);
                waypoints_accel = Eigen::VectorXd::Ones(3 * num_penalized_steps);
                u_0 = Eigen::VectorXd::Ones(3);
            }
        };

        struct LagrangeMultipliers {
            Eigen::VectorXd eq; // equality constraint residuals
            Eigen::VectorXd pos; // position constraint residuals
            Eigen::VectorXd waypoints_pos; // waypoint constraint residuals
            Eigen::VectorXd waypoints_vel; // waypoint constraint residuals
            Eigen::VectorXd waypoints_accel; // acceleration constraint residuals --> to do change this
            Eigen::VectorXd u_0;
            Eigen::VectorXd u_dot_0;
            Eigen::VectorXd u_ddot_0;

            LagrangeMultipliers(int j, int K, int num_penalized_steps) {
                eq = Eigen::VectorXd::Zero((2 + j) * 3 * K);
                pos = Eigen::VectorXd::Zero(6 * K);
                waypoints_pos = Eigen::VectorXd::Zero(3 * num_penalized_steps);
                waypoints_vel = Eigen::VectorXd::Zero(3 * num_penalized_steps);
                waypoints_accel = Eigen::VectorXd::Zero(3 * num_penalized_steps);
                u_0 = Eigen::VectorXd::Zero(3);
                u_dot_0 = Eigen::VectorXd::Zero(3);
                u_ddot_0 = Eigen::VectorXd::Zero(3);
            }
        };

        struct CostMatrices {
            Eigen::SparseMatrix<double> Q, q; // why is q a matrix? --> clarify this later
            Eigen::SparseMatrix<double> A_check_const_terms;
            Eigen::SparseMatrix<double> A_check;
            Eigen::SparseVector<double> b_check;
        };

        
        // Private variables
        Eigen::SparseMatrix<double> W, W_dot, W_ddot;
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
        Eigen::MatrixXd waypoints;
        Eigen::VectorXd initial_pos;
        Eigen::SparseMatrix<double> collision_envelope; // this drone's collision envelope - NOT the other obstacles' collision envelopes

        // Private methods
        ConstSelectionMatrices constSelectionMatrices;
        void initConstSelectionMatrices(); // to do remove this for a default constructor


        Eigen::MatrixXd extractWaypointsInCurrentHorizon(const double t,
                                                        const Eigen::MatrixXd& waypoints);
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
                                    CostMatrices& costMatrices,
                                    bool waypoint_position_constraints,
                                    bool waypoint_velocity_constraints,
                                    bool waypoint_acceleration_constraints,
                                    Eigen::VectorXd u_0_prev,
                                    Eigen::VectorXd u_dot_0_prev,
                                    Eigen::VectorXd u_ddot_0_prev);

        void initVariableSelectionMatrices(int j, Eigen::VectorXd& penalized_steps,
                            VariableSelectionMatrices& variableSelectionMatrices);
                            
        void initOptimizationVariables(int j, Eigen::VectorXd& alpha,
                                    Eigen::VectorXd& beta, Eigen::VectorXd& d,
                                    Eigen::VectorXd& zeta_1);

        void initConstConstraintMatrices(int j, Eigen::VectorXd x_0,
                                Eigen::VectorXd xi,
                                Eigen::SparseMatrix<double>& S_theta,
                                Eigen::MatrixXd& extracted_waypoints,
                                VariableSelectionMatrices& variableSelectionMatrices,
                                Constraints& constraints);

        void initCostMatrices(Eigen::VectorXd& penalized_steps,
                            Eigen::VectorXd x_0,
                            Eigen::SparseMatrix<double>& X_g,
                            CostMatrices& costMatrices,
                            Eigen::VectorXd u_0_prev,
                            Eigen::VectorXd u_dot_0_prev,
                            Eigen::VectorXd u_ddot_0_prev);

        void computeX_g(Eigen::MatrixXd& extracted_waypoints,
                        Eigen::VectorXd& penalized_steps,
                        Eigen::SparseMatrix<double>& X_g);

        void computeZeta1(int iters, double rho,
                        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
                        CostMatrices& costMatrices,
                        Constraints& constraints,
                        Eigen::VectorXd& s,
                        LagrangeMultipliers& lambda,
                        Eigen::VectorXd& zeta_1,
                        bool waypoint_position_constraints,
                        bool waypoint_velocity_constraints,
                        bool waypoint_acceleration_constraints,
                        Eigen::VectorXd& u_0_prev);

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
                            Residuals& residuals,
                            Eigen::VectorXd& u_0_prev);

        void updateLagrangeMultipliers(double rho, Residuals& residuals, LagrangeMultipliers& lambda);

        DroneResult computeDroneResult(double current_time, Eigen::VectorXd& zeta_1,Eigen::VectorXd x_0);

        void printUnsatisfiedResiduals(const Residuals& residuals,
                                        double threshold,
                                        bool waypoint_position_constraints,
                                        bool waypoint_velocity_constraints,
                                        bool waypoint_acceleration_constraints);

        Eigen::VectorXd U_to_zeta_1(Eigen::VectorXd& U);
};

#endif
