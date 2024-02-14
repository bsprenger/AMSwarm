#ifndef DRONE_H
#define DRONE_H

#include <utils.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>


using namespace Eigen;


class Drone {
    public:
        // Public struct definitions

        class DroneResult {
            public:
                VectorXd position_state_time_stamps; // time stamps for both position and state
                VectorXd control_input_time_stamps;

                MatrixXd position_trajectory;
                MatrixXd state_trajectory;
                MatrixXd control_input_trajectory;

                VectorXd position_trajectory_vector;
                VectorXd state_trajectory_vector;
                VectorXd control_input_trajectory_vector;

                VectorXd spline_coeffs;

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

        struct Dynamics {
            MatrixXd A, B, A_prime, B_prime;
        };

        struct SparseDynamics {
            SparseMatrix<double> A, B, A_prime, B_prime;

            SparseDynamics() {}
            
            SparseDynamics(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, 
                   const Eigen::SparseMatrix<double>& A_prime, const Eigen::SparseMatrix<double>& B_prime) 
            : A(A), B(B), A_prime(A_prime), B_prime(B_prime) {}
        };

        // Constructors
        Drone(MatrixXd waypoints, // necessary input
                MPCConfig config,
                MPCWeights weights,
                PhysicalLimits limits,
                SparseDynamics dynamics,
                VectorXd initial_pos = VectorXd::Zero(3));

        // Public methods
        DroneResult solve(const double current_time,
                                const VectorXd x_0,
                                const int j,
                                std::vector<SparseMatrix<double>> thetas,
                                const VectorXd xi,
                                SolveOptions& opt,
                                const VectorXd& initial_guess = VectorXd());
        
        // Getters
        VectorXd getInitialPosition();
        SparseMatrix<double> getCollisionEnvelope();
        MatrixXd getWaypoints();
        float getDeltaT();
        int getK();

        // Setters
        // To do


    private:
        // Private struct definitions 
        struct ConstSelectionMatrices {
            SparseMatrix<double> M_p, M_v, M_a; // maybe rename to pos,vel,acc

            ConstSelectionMatrices(int K) {
                // Intermediate matrices used in building selection matrices
                SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
                SparseMatrix<double> eyeK = utils::getSparseIdentity(K);
                SparseMatrix<double> zeroMat(3, 3);
                zeroMat.setZero();

                M_p = utils::kroneckerProduct(eyeK, utils::horzcat(eye3, zeroMat));
                M_v = utils::kroneckerProduct(eyeK, utils::horzcat(zeroMat, eye3));
                M_a = utils::kroneckerProduct(eyeK, utils::horzcat(zeroMat, eye3));
            }
        };

        struct VariableSelectionMatrices {
            SparseMatrix<double> M_x, M_y, M_z, M_waypoints_position, M_waypoints_velocity; // maybe rename to x,y,z,timestep?

            VariableSelectionMatrices(int K, int j, VectorXd& penalized_steps) {
                SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
                SparseMatrix<double> eye6 = utils::getSparseIdentity(6);
                SparseMatrix<double> eyeK = utils::getSparseIdentity(K);
                SparseMatrix<double> eyeK2j = utils::getSparseIdentity((2 + j) * K);
                SparseMatrix<double> zeroMat(3, 3);
                zeroMat.setZero();
                SparseMatrix<double> x_step(1, 3);
                x_step.coeffRef(0, 0) = 1.0;
                SparseMatrix<double> y_step(1, 3);
                y_step.coeffRef(0, 1) = 1.0;
                SparseMatrix<double> z_step(1, 3);
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
            SparseMatrix<double> G_eq, G_pos, G_waypoints_pos,G_waypoints_vel, G_waypoints_accel;
            VectorXd h_eq, h_pos, h_waypoints_pos, h_waypoints_vel, h_waypoints_accel;
            VectorXd c_eq, c_waypoints_pos, c_waypoints_vel, c_waypoints_accel;

            // Constructor
            Constraints(const Drone* parentDrone, int j, const VectorXd& x_0,
                        const VectorXd& xi, std::vector<SparseMatrix<double>> thetas,
                        const MatrixXd& extracted_waypoints,
                        const VariableSelectionMatrices& variableSelectionMatrices) {
                            G_waypoints_pos = variableSelectionMatrices.M_waypoints_position * parentDrone->S_u * parentDrone->W_input;
                            c_waypoints_pos = variableSelectionMatrices.M_waypoints_position * parentDrone->S_x * x_0;
                            h_waypoints_pos = extracted_waypoints.block(0, 1, extracted_waypoints.rows(), extracted_waypoints.cols() - 4).reshaped<RowMajor>();

                            G_waypoints_vel = variableSelectionMatrices.M_waypoints_velocity * parentDrone->S_u * parentDrone->W_input;
                            c_waypoints_vel = variableSelectionMatrices.M_waypoints_velocity * parentDrone->S_x * x_0;
                            h_waypoints_vel = extracted_waypoints.block(0, 4, extracted_waypoints.rows(), extracted_waypoints.cols() - 4).reshaped<RowMajor>();

                            G_waypoints_accel = variableSelectionMatrices.M_waypoints_position * parentDrone->S_u_prime * parentDrone->W_input;
                            c_waypoints_accel = variableSelectionMatrices.M_waypoints_position * parentDrone->S_x_prime * x_0;
                            h_waypoints_accel = VectorXd::Zero(G_waypoints_accel.rows());

                            SparseMatrix<double> S_theta = utils::blkDiag(thetas);
                            SparseMatrix<double> G_eq_blk1 = parentDrone->constSelectionMatrices.M_v * parentDrone->S_u * parentDrone->W_input;
                            SparseMatrix<double> G_eq_blk2 = parentDrone->constSelectionMatrices.M_a * parentDrone->S_u_prime * parentDrone->W_input;
                            SparseMatrix<double> G_eq_blk3 = S_theta * utils::replicateSparseMatrix(parentDrone->constSelectionMatrices.M_p * parentDrone->S_u * parentDrone->W_input, j, 1);
                            G_eq = utils::vertcat(utils::vertcat(G_eq_blk1, G_eq_blk2), G_eq_blk3);

                            SparseMatrix<double> G_pos_blk1 = parentDrone->constSelectionMatrices.M_p * parentDrone->S_u * parentDrone->W_input;
                            SparseMatrix<double> G_pos_blk2 = -parentDrone->constSelectionMatrices.M_p * parentDrone->S_u * parentDrone->W_input;
                            G_pos = utils::vertcat(G_pos_blk1, G_pos_blk2);

                            VectorXd h_pos_blk1 = parentDrone->limits.p_max.replicate(parentDrone->config.K, 1) - parentDrone->constSelectionMatrices.M_p * parentDrone->S_x * x_0;
                            VectorXd h_pos_blk2 = -parentDrone->limits.p_min.replicate(parentDrone->config.K, 1) + parentDrone->constSelectionMatrices.M_p * parentDrone->S_x * x_0;
                            h_pos.resize(h_pos_blk1.rows() + h_pos_blk2.rows());
                            h_pos << h_pos_blk1, h_pos_blk2;

                            VectorXd c_eq_blk1 = parentDrone->constSelectionMatrices.M_v * parentDrone->S_x * x_0;
                            VectorXd c_eq_blk2 = parentDrone->constSelectionMatrices.M_a * parentDrone->S_x_prime * x_0;
                            VectorXd c_eq_blk3 = S_theta * ((parentDrone->constSelectionMatrices.M_p * parentDrone->S_x * x_0).replicate(j, 1) - xi);
                            c_eq.resize(c_eq_blk1.rows() + c_eq_blk2.rows() + c_eq_blk3.rows());
                            c_eq << c_eq_blk1, c_eq_blk2, c_eq_blk3;
                        }
        };

        struct Residuals {
            VectorXd eq; // equality constraint residuals
            VectorXd pos; // position constraint residuals
            VectorXd waypoints_pos; // waypoint constraint residuals
            VectorXd waypoints_vel; // waypoint constraint residuals
            VectorXd waypoints_accel; // acceleration constraint residuals --> to do change this
            VectorXd input_continuity; // input continuity constraint residuals
            VectorXd input_dot_continuity; // input continuity constraint residuals
            VectorXd input_ddot_continuity; // input continuity constraint residuals

            Residuals(const Drone* parentDrone, int j, int K, int num_penalized_steps) {
                eq = VectorXd::Ones((2 + j) * 3 * K); // TODO something more intelligent then setting these to 1 -> they should be bigger than threshold
                pos = VectorXd::Ones(6 * K);
                waypoints_pos = VectorXd::Ones(3 * num_penalized_steps);
                waypoints_vel = VectorXd::Ones(3 * num_penalized_steps);
                waypoints_accel = VectorXd::Ones(3 * num_penalized_steps);

                input_continuity = VectorXd::Ones(3);
                input_dot_continuity = VectorXd::Ones(3);
                input_ddot_continuity = VectorXd::Ones(3);
            }
        };

        struct LagrangeMultipliers {
            VectorXd eq; // equality constraint residuals
            VectorXd pos; // position constraint residuals
            VectorXd waypoints_pos; // waypoint constraint residuals
            VectorXd waypoints_vel; // waypoint constraint residuals
            VectorXd waypoints_accel; // acceleration constraint residuals --> to do change this
            VectorXd input_continuity; // input continuity constraint residuals
            VectorXd input_dot_continuity;
            VectorXd input_ddot_continuity;

            LagrangeMultipliers(const Drone* parentDrone, int j, int K, int num_penalized_steps) {
                eq = VectorXd::Zero((2 + j) * 3 * K);
                pos = VectorXd::Zero(6 * K);
                waypoints_pos = VectorXd::Zero(3 * num_penalized_steps);
                waypoints_vel = VectorXd::Zero(3 * num_penalized_steps);
                waypoints_accel = VectorXd::Zero(3 * num_penalized_steps);

                input_continuity = VectorXd::Zero(3);
                input_dot_continuity = VectorXd::Zero(3);
                input_ddot_continuity = VectorXd::Zero(3);
            }
        };

        struct CostMatrices {
            SparseMatrix<double> Q, q; // why is q a matrix? --> clarify this later
            SparseMatrix<double> A_check_const_terms;
            SparseMatrix<double> A_check;
            SparseVector<double> b_check;

            CostMatrices(const Drone* parentDrone,
                        VectorXd& penalized_steps,
                        VectorXd x_0,
                        SparseMatrix<double>& X_g,
                        VectorXd u_0_prev,
                        VectorXd u_dot_0_prev,
                        VectorXd u_ddot_0_prev,
                        Constraints& constraints,
                        SolveOptions& opt) {

                SparseMatrix<double> eye3 = utils::getSparseIdentity(3);
                SparseMatrix<double> eyeK = utils::getSparseIdentity(parentDrone->config.K);

                std::vector<SparseMatrix<double>> tmp_cost_vec = {eye3 * parentDrone->weights.w_goal_pos, eye3 * parentDrone->weights.w_goal_vel}; // clarify this
                SparseMatrix<double> R_g = utils::blkDiag(tmp_cost_vec); // clarify this

                SparseMatrix<double> tmp_R_g_tilde(parentDrone->config.K,parentDrone->config.K); // for selecting which steps to penalize
                for (int idx : penalized_steps) {
                    tmp_R_g_tilde.insert(idx - 1, idx - 1) = 1.0; // this needs to be clarified -> since the first block in R_g_tilde corresponds to x(1), we need to subtract 1 from the index. penalized_steps gives the TIME STEP number, not matrix index
                }
                SparseMatrix<double> R_g_tilde = utils::kroneckerProduct(tmp_R_g_tilde, R_g);
                
                // initialize R_s_tilde
                SparseMatrix<double> R_s = eye3 * parentDrone->weights.w_smoothness;
                SparseMatrix<double> R_s_tilde = utils::kroneckerProduct(eyeK, R_s);

                // initialize cost matrices
                Q = 2.0 * parentDrone->W_input.transpose() * parentDrone->S_u.transpose() * R_g_tilde * parentDrone->S_u * parentDrone->W_input
                                + 2.0 * parentDrone->W_input.transpose() * parentDrone->S_u_prime.transpose() * parentDrone->constSelectionMatrices.M_a.transpose() * R_s_tilde * parentDrone->constSelectionMatrices.M_a * parentDrone->S_u_prime * parentDrone->W_input
                                + 2.0 * parentDrone->weights.w_input_smoothness * parentDrone->W_ddot.transpose() * parentDrone->W_ddot
                                + 2.0 * parentDrone->weights.w_input_continuity * parentDrone->W.block(0,0,3,3*(parentDrone->config.n+1)).transpose() * parentDrone->W.block(0,0,3,3*(parentDrone->config.n+1))
                                + 2.0 * parentDrone->weights.w_input_dot_continuity * parentDrone->W_dot.block(0,0,3,3*(parentDrone->config.n+1)).transpose() * parentDrone->W_dot.block(0,0,3,3*(parentDrone->config.n+1))
                                + 2.0 * parentDrone->weights.w_input_ddot_continuity * parentDrone->W_ddot.block(0,0,3,3*(parentDrone->config.n+1)).transpose() * parentDrone->W_ddot.block(0,0,3,3*(parentDrone->config.n+1));
                q = 2.0 * parentDrone->W_input.transpose() * parentDrone->S_u.transpose() * R_g_tilde.transpose() * (parentDrone->S_x * x_0 - X_g)
                                + 2.0 * parentDrone->W_input.transpose() * parentDrone->S_u_prime.transpose() * parentDrone->constSelectionMatrices.M_a.transpose() * R_s_tilde * parentDrone->constSelectionMatrices.M_a * parentDrone->S_x_prime * x_0
                                - 2.0 * parentDrone->weights.w_input_continuity * parentDrone->W.block(0,0,3,3*(parentDrone->config.n+1)).transpose() * u_0_prev
                                - 2.0 * parentDrone->weights.w_input_dot_continuity * parentDrone->W_dot.block(0,0,3,3*(parentDrone->config.n+1)).transpose() * u_dot_0_prev
                                - 2.0 * parentDrone->weights.w_input_ddot_continuity * parentDrone->W_ddot.block(0,0,3,3*(parentDrone->config.n+1)).transpose() * u_ddot_0_prev;

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
                    A_check_const_terms += parentDrone->W.block(0,0,3,3*(parentDrone->config.n+1)).transpose() * parentDrone->W.block(0,0,3,3*(parentDrone->config.n+1));
                }
                if (opt.input_dot_continuity_constraints) {
                    A_check_const_terms += parentDrone->W_dot.block(0,0,3,3*(parentDrone->config.n+1)).transpose() * parentDrone->W_dot.block(0,0,3,3*(parentDrone->config.n+1));
                }
                if (opt.input_ddot_continuity_constraints) {
                    A_check_const_terms += parentDrone->W_ddot.block(0,0,3,3*(parentDrone->config.n+1)).transpose() * parentDrone->W_ddot.block(0,0,3,3*(parentDrone->config.n+1));
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

        void computeX_g(MatrixXd& extracted_waypoints,
                        VectorXd& penalized_steps,
                        SparseMatrix<double>& X_g);

        void compute_h_eq(int, VectorXd&, VectorXd&, VectorXd&,VectorXd&);

        void compute_d(int j, double rho,
                    Constraints& constraints, VectorXd& zeta_1,
                    LagrangeMultipliers& lambda,
                    VectorXd& alpha, VectorXd& beta,
                    VectorXd& d);

        void computeAlphaBeta(double rho, Constraints& constraints,
                            VectorXd& zeta_1,
                            LagrangeMultipliers& lambda, 
                            VectorXd& alpha, VectorXd& beta,
                            VariableSelectionMatrices& variableSelectionMatrices);
                            
        void computeResiduals(Constraints& constraints,
                            VectorXd& zeta_1, VectorXd& s,
                            Residuals& residuals,
                            VectorXd& u_0_prev,
                            VectorXd& u_dot_0_prev,
                            VectorXd& u_ddot_0_prev);

        void updateLagrangeMultipliers(double rho, Residuals& residuals, LagrangeMultipliers& lambda);

        void updateCostMatrices(double rho, CostMatrices& costMatrices,
                            Constraints& constraints,
                            VectorXd& s,
                            LagrangeMultipliers& lambda,
                            SolveOptions& opt,
                            VectorXd& u_0_prev,
                            VectorXd& u_dot_0_prev,
                            VectorXd& u_ddot_0_prev);

        DroneResult computeDroneResult(double current_time, VectorXd& zeta_1,VectorXd x_0);

        void printUnsatisfiedResiduals(const Residuals& residuals,
                                        SolveOptions& opt);

        VectorXd U_to_zeta_1(const VectorXd& U);
};

#endif
