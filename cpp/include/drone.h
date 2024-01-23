#ifndef DRONE_H
#define DRONE_H

#include <utils.h>

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
            double waypoint_velocity_threshold = 0.01;
            double waypoint_acceleration_threshold = 0.01;
            double input_continuity_threshold = 0.01;
            double input_dot_continuity_threshold = 0.01;
            double input_ddot_continuity_threshold = 0.01;
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
                float w_goal_pos = 7000,
                float w_goal_vel = 1000,
                float w_smoothness = 100,
                float w_input_smoothness = 1000,
                float w_input_continuity = 100,
                float w_input_dot_continuity = 100,
                float w_input_ddot_continuity = 100,
                float v_bar = 1.73,
                float f_bar = 1.5*9.8);

        // Public methods
        DroneResult solve(const double current_time,
                                const Eigen::VectorXd x_0,
                                Eigen::VectorXd& initial_guess_control_input_trajectory_vector,
                                const int j,
                                std::vector<Eigen::SparseMatrix<double>> thetas,
                                const Eigen::VectorXd xi,
                                SolveOptions& opt);
        
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
            Eigen::SparseMatrix<double> M_p, M_v, M_a; // maybe rename to pos,vel,acc

            ConstSelectionMatrices(int K) {
                // Intermediate matrices used in building selection matrices
                Eigen::SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
                Eigen::SparseMatrix<double> eyeK = utils::getSparseIdentity(K);
                Eigen::SparseMatrix<double> zeroMat(3, 3);
                zeroMat.setZero();

                M_p = utils::kroneckerProduct(eyeK, utils::horzcat(eye3, zeroMat));
                M_v = utils::kroneckerProduct(eyeK, utils::horzcat(zeroMat, eye3));
                M_a = utils::kroneckerProduct(eyeK, utils::horzcat(zeroMat, eye3));
            }
        };

        struct VariableSelectionMatrices {
            Eigen::SparseMatrix<double> M_x, M_y, M_z, M_waypoints_position, M_waypoints_velocity; // maybe rename to x,y,z,timestep?

            VariableSelectionMatrices(int K, int j, Eigen::VectorXd& penalized_steps) {
                Eigen::SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
                Eigen::SparseMatrix<double> eye6 = utils::getSparseIdentity(6);
                Eigen::SparseMatrix<double> eyeK = utils::getSparseIdentity(K);
                Eigen::SparseMatrix<double> eyeK2j = utils::getSparseIdentity((2 + j) * K);
                Eigen::SparseMatrix<double> zeroMat(3, 3);
                zeroMat.setZero();
                Eigen::SparseMatrix<double> x_step(1, 3);
                x_step.coeffRef(0, 0) = 1.0;
                Eigen::SparseMatrix<double> y_step(1, 3);
                y_step.coeffRef(0, 1) = 1.0;
                Eigen::SparseMatrix<double> z_step(1, 3);
                z_step.coeffRef(0, 2) = 1.0;

                M_x = utils::kroneckerProduct(eyeK2j, x_step);
                M_y = utils::kroneckerProduct(eyeK2j, y_step);
                M_z = utils::kroneckerProduct(eyeK2j, z_step);

                M_waypoints_position.resize(3 * penalized_steps.size(), 6 * K);
                for (int i = 0; i < penalized_steps.size(); ++i) {
                    utils::replaceSparseBlock(M_waypoints_position, eye3, 3 * i, 6 * (penalized_steps(i) - 1));
                }

                M_waypoints_velocity.resize(3 * penalized_steps.size(), 6 * K);
                for (int i = 0; i < penalized_steps.size(); ++i) {
                    utils::replaceSparseBlock(M_waypoints_velocity, eye3, 3 * i, 6 * (penalized_steps(i) - 1) + 3);
                }
            }
        };

        struct Constraints {
            Eigen::SparseMatrix<double> G_eq, G_pos, G_waypoints_pos,G_waypoints_vel, G_waypoints_accel;
            Eigen::VectorXd h_eq, h_pos, h_waypoints_pos, h_waypoints_vel, h_waypoints_accel;
            Eigen::VectorXd c_eq, c_waypoints_pos, c_waypoints_vel, c_waypoints_accel;

            // Constructor
            Constraints(const Drone* parentDrone, int j, const Eigen::VectorXd& x_0,
                        const Eigen::VectorXd& xi, std::vector<Eigen::SparseMatrix<double>> thetas,
                        const Eigen::MatrixXd& extracted_waypoints,
                        const VariableSelectionMatrices& variableSelectionMatrices) {
                            G_waypoints_pos = variableSelectionMatrices.M_waypoints_position * parentDrone->S_u * parentDrone->W;
                            c_waypoints_pos = variableSelectionMatrices.M_waypoints_position * parentDrone->S_x * x_0;
                            h_waypoints_pos = extracted_waypoints.block(0, 1, extracted_waypoints.rows(), extracted_waypoints.cols() - 4).reshaped<Eigen::RowMajor>();

                            G_waypoints_vel = variableSelectionMatrices.M_waypoints_velocity * parentDrone->S_u * parentDrone->W;
                            c_waypoints_vel = variableSelectionMatrices.M_waypoints_velocity * parentDrone->S_x * x_0;
                            h_waypoints_vel = extracted_waypoints.block(0, 4, extracted_waypoints.rows(), extracted_waypoints.cols() - 4).reshaped<Eigen::RowMajor>();

                            G_waypoints_accel = variableSelectionMatrices.M_waypoints_position * parentDrone->S_u_prime * parentDrone->W;
                            c_waypoints_accel = variableSelectionMatrices.M_waypoints_position * parentDrone->S_x_prime * x_0;
                            h_waypoints_accel = Eigen::VectorXd::Zero(G_waypoints_accel.rows());

                            Eigen::SparseMatrix<double> S_theta = utils::blkDiag(thetas);
                            Eigen::SparseMatrix<double> G_eq_blk1 = parentDrone->constSelectionMatrices.M_v * parentDrone->S_u * parentDrone->W;
                            Eigen::SparseMatrix<double> G_eq_blk2 = parentDrone->constSelectionMatrices.M_a * parentDrone->S_u_prime * parentDrone->W;
                            Eigen::SparseMatrix<double> G_eq_blk3 = S_theta * utils::replicateSparseMatrix(parentDrone->constSelectionMatrices.M_p * parentDrone->S_u * parentDrone->W, j, 1);
                            G_eq = utils::vertcat(utils::vertcat(G_eq_blk1, G_eq_blk2), G_eq_blk3);

                            Eigen::SparseMatrix<double> G_pos_blk1 = parentDrone->constSelectionMatrices.M_p * parentDrone->S_u * parentDrone->W;
                            Eigen::SparseMatrix<double> G_pos_blk2 = -parentDrone->constSelectionMatrices.M_p * parentDrone->S_u * parentDrone->W;
                            G_pos = utils::vertcat(G_pos_blk1, G_pos_blk2);

                            Eigen::VectorXd h_pos_blk1 = parentDrone->p_max.replicate(parentDrone->K, 1) - parentDrone->constSelectionMatrices.M_p * parentDrone->S_x * x_0;
                            Eigen::VectorXd h_pos_blk2 = -parentDrone->p_min.replicate(parentDrone->K, 1) + parentDrone->constSelectionMatrices.M_p * parentDrone->S_x * x_0;
                            h_pos.resize(h_pos_blk1.rows() + h_pos_blk2.rows());
                            h_pos << h_pos_blk1, h_pos_blk2;

                            Eigen::VectorXd c_eq_blk1 = parentDrone->constSelectionMatrices.M_v * parentDrone->S_x * x_0;
                            Eigen::VectorXd c_eq_blk2 = parentDrone->constSelectionMatrices.M_a * parentDrone->S_x_prime * x_0;
                            Eigen::VectorXd c_eq_blk3 = S_theta * ((parentDrone->constSelectionMatrices.M_p * parentDrone->S_x * x_0).replicate(j, 1) - xi);
                            c_eq.resize(c_eq_blk1.rows() + c_eq_blk2.rows() + c_eq_blk3.rows());
                            c_eq << c_eq_blk1, c_eq_blk2, c_eq_blk3;
                        }
        };

        struct Residuals {
            Eigen::VectorXd eq; // equality constraint residuals
            Eigen::VectorXd pos; // position constraint residuals
            Eigen::VectorXd waypoints_pos; // waypoint constraint residuals
            Eigen::VectorXd waypoints_vel; // waypoint constraint residuals
            Eigen::VectorXd waypoints_accel; // acceleration constraint residuals --> to do change this
            Eigen::VectorXd input_continuity; // input continuity constraint residuals
            Eigen::VectorXd input_dot_continuity; // input continuity constraint residuals
            Eigen::VectorXd input_ddot_continuity; // input continuity constraint residuals

            Residuals(int j, int K, int num_penalized_steps) {
                eq = Eigen::VectorXd::Ones((2 + j) * 3 * K); // TODO something more intelligent then setting these to 1 -> they should be bigger than threshold
                pos = Eigen::VectorXd::Ones(6 * K);
                waypoints_pos = Eigen::VectorXd::Ones(3 * num_penalized_steps);
                waypoints_vel = Eigen::VectorXd::Ones(3 * num_penalized_steps);
                waypoints_accel = Eigen::VectorXd::Ones(3 * num_penalized_steps);
                input_continuity = Eigen::VectorXd::Ones(3);
                input_dot_continuity = Eigen::VectorXd::Ones(3);
                input_ddot_continuity = Eigen::VectorXd::Ones(3);
            }
        };

        struct LagrangeMultipliers {
            Eigen::VectorXd eq; // equality constraint residuals
            Eigen::VectorXd pos; // position constraint residuals
            Eigen::VectorXd waypoints_pos; // waypoint constraint residuals
            Eigen::VectorXd waypoints_vel; // waypoint constraint residuals
            Eigen::VectorXd waypoints_accel; // acceleration constraint residuals --> to do change this
            Eigen::VectorXd input_continuity; // input continuity constraint residuals
            Eigen::VectorXd input_dot_continuity;
            Eigen::VectorXd input_ddot_continuity;

            LagrangeMultipliers(int j, int K, int num_penalized_steps) {
                eq = Eigen::VectorXd::Zero((2 + j) * 3 * K);
                pos = Eigen::VectorXd::Zero(6 * K);
                waypoints_pos = Eigen::VectorXd::Zero(3 * num_penalized_steps);
                waypoints_vel = Eigen::VectorXd::Zero(3 * num_penalized_steps);
                waypoints_accel = Eigen::VectorXd::Zero(3 * num_penalized_steps);
                input_continuity = Eigen::VectorXd::Zero(3);
                input_dot_continuity = Eigen::VectorXd::Zero(3);
                input_ddot_continuity = Eigen::VectorXd::Zero(3);
            }
        };

        struct CostMatrices {
            Eigen::SparseMatrix<double> Q, q; // why is q a matrix? --> clarify this later
            Eigen::SparseMatrix<double> A_check_const_terms;
            Eigen::SparseMatrix<double> A_check;
            Eigen::SparseVector<double> b_check;

            CostMatrices(const Drone* parentDrone,
                        Eigen::VectorXd& penalized_steps,
                        Eigen::VectorXd x_0,
                        Eigen::SparseMatrix<double>& X_g,
                        Eigen::VectorXd u_0_prev,
                        Eigen::VectorXd u_dot_0_prev,
                        Eigen::VectorXd u_ddot_0_prev,
                        Constraints& constraints,
                        SolveOptions& opt) {
                Eigen::SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
                Eigen::SparseMatrix<double> eyeK = utils::getSparseIdentity(parentDrone->K);

                std::vector<Eigen::SparseMatrix<double>> tmp_cost_vec = {eye3 * parentDrone->w_goal_pos, eye3 * parentDrone->w_goal_vel}; // clarify this
                Eigen::SparseMatrix<double> R_g = utils::blkDiag(tmp_cost_vec); // clarify this

                Eigen::SparseMatrix<double> tmp_R_g_tilde(parentDrone->K,parentDrone->K); // for selecting which steps to penalize
                for (int idx : penalized_steps) {
                    tmp_R_g_tilde.insert(idx - 1, idx - 1) = 1.0; // this needs to be clarified -> since the first block in R_g_tilde corresponds to x(1), we need to subtract 1 from the index. penalized_steps gives the TIME STEP number, not matrix index
                }
                Eigen::SparseMatrix<double> R_g_tilde = utils::kroneckerProduct(tmp_R_g_tilde, R_g);
                
                // initialize R_s_tilde
                Eigen::SparseMatrix<double> R_s = eye3 * parentDrone->w_smoothness;
                Eigen::SparseMatrix<double> R_s_tilde = utils::kroneckerProduct(eyeK, R_s);

                // initialize cost matrices
                Q = 2.0 * parentDrone->W.transpose() * parentDrone->S_u.transpose() * R_g_tilde * parentDrone->S_u * parentDrone->W
                                + 2.0 * parentDrone->W.transpose() * parentDrone->S_u_prime.transpose() * parentDrone->constSelectionMatrices.M_a.transpose() * R_s_tilde * parentDrone->constSelectionMatrices.M_a * parentDrone->S_u_prime * parentDrone->W
                                + 2.0 * parentDrone->w_input_smoothness * parentDrone->W_ddot.transpose() * parentDrone->W_ddot
                                + 2.0 * parentDrone->w_input_continuity * parentDrone->W.block(0,0,3,3*(parentDrone->n+1)).transpose() * parentDrone->W.block(0,0,3,3*(parentDrone->n+1))
                                + 2.0 * parentDrone->w_input_dot_continuity * parentDrone->W_dot.block(0,0,3,3*(parentDrone->n+1)).transpose() * parentDrone->W_dot.block(0,0,3,3*(parentDrone->n+1))
                                + 2.0 * parentDrone->w_input_ddot_continuity * parentDrone->W_ddot.block(0,0,3,3*(parentDrone->n+1)).transpose() * parentDrone->W_ddot.block(0,0,3,3*(parentDrone->n+1));
                q = 2.0 * parentDrone->W.transpose() * parentDrone->S_u.transpose() * R_g_tilde.transpose() * (parentDrone->S_x * x_0 - X_g)
                                + 2.0 * parentDrone->W.transpose() * parentDrone->S_u_prime.transpose() * parentDrone->constSelectionMatrices.M_a.transpose() * R_s_tilde * parentDrone->constSelectionMatrices.M_a * parentDrone->S_x_prime * x_0
                                - 2.0 * parentDrone->w_input_continuity * parentDrone->W.block(0,0,3,3*(parentDrone->n+1)).transpose() * u_0_prev
                                - 2.0 * parentDrone->w_input_dot_continuity * parentDrone->W_dot.block(0,0,3,3*(parentDrone->n+1)).transpose() * u_dot_0_prev
                                - 2.0 * parentDrone->w_input_ddot_continuity * parentDrone->W_ddot.block(0,0,3,3*(parentDrone->n+1)).transpose() * u_ddot_0_prev;

                A_check_const_terms = constraints.G_eq.transpose() * constraints.G_eq + constraints.G_pos.transpose() * constraints.G_pos;

                // TODO look at what needs to be moved to constraints struct
                if (opt.waypoint_position_constraints) {
                    A_check_const_terms += constraints.G_waypoints_pos.transpose() * constraints.G_waypoints_pos;
                }
                if (opt.waypoint_velocity_constraints) {
                    A_check_const_terms += constraints.G_waypoints_vel.transpose() * constraints.G_waypoints_vel;
                }
                if (opt.waypoint_acceleration_constraints) {
                    A_check_const_terms += constraints.G_waypoints_accel.transpose() * constraints.G_waypoints_accel;
                }
                if (opt.input_continuity_constraints) {
                    A_check_const_terms += parentDrone->W.block(0,0,3,3*(parentDrone->n+1)).transpose() * parentDrone->W.block(0,0,3,3*(parentDrone->n+1));
                }
                if (opt.input_dot_continuity_constraints) {
                    A_check_const_terms += parentDrone->W_dot.block(0,0,3,3*(parentDrone->n+1)).transpose() * parentDrone->W_dot.block(0,0,3,3*(parentDrone->n+1));
                }
                if (opt.input_ddot_continuity_constraints) {
                    A_check_const_terms += parentDrone->W_ddot.block(0,0,3,3*(parentDrone->n+1)).transpose() * parentDrone->W_ddot.block(0,0,3,3*(parentDrone->n+1));
                }
            }
        };

        
        // Private variables
        Eigen::SparseMatrix<double> W, W_dot, W_ddot;
        Eigen::SparseMatrix<double> S_x, S_u, S_x_prime, S_u_prime;
        int K;
        int n;
        float delta_t;
        float t_f;
        
        // cost weights
        float w_goal_pos;
        float w_goal_vel;
        float w_smoothness;
        float w_input_smoothness;
        float w_input_continuity;
        float w_input_dot_continuity;
        float w_input_ddot_continuity;

        // physical limits
        Eigen::VectorXd p_min;
        Eigen::VectorXd p_max;
        double v_bar;
        double f_bar;

        Eigen::MatrixXd waypoints;
        Eigen::VectorXd initial_pos;
        Eigen::SparseMatrix<double> collision_envelope; // this drone's collision envelope - NOT the other obstacles' collision envelopes

        // Private methods
        ConstSelectionMatrices constSelectionMatrices;


        Eigen::MatrixXd extractWaypointsInCurrentHorizon(const double t,
                                                        const Eigen::MatrixXd& waypoints);
        void generateBernsteinMatrices();
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> loadDynamicsMatricesFromFile(const std::string&);
        std::tuple<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> loadSparseDynamicsMatricesFromFile(const std::string&);
        void generateFullHorizonDynamicsMatrices(std::string&);

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
                        SolveOptions& opt,
                        Eigen::VectorXd& u_0_prev,
                        Eigen::VectorXd& u_dot_0_prev,
                        Eigen::VectorXd& u_ddot_0_prev);

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
                            Eigen::VectorXd& u_0_prev,
                            Eigen::VectorXd& u_dot_0_prev,
                            Eigen::VectorXd& u_ddot_0_prev);

        void updateLagrangeMultipliers(double rho, Residuals& residuals, LagrangeMultipliers& lambda);

        DroneResult computeDroneResult(double current_time, Eigen::VectorXd& zeta_1,Eigen::VectorXd x_0);

        void printUnsatisfiedResiduals(const Residuals& residuals,
                                        SolveOptions& opt);

        Eigen::VectorXd U_to_zeta_1(Eigen::VectorXd& U);
};

#endif
