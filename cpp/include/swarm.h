#ifndef SWARM_H
#define SWARM_H

#include <vector>
#include <drone.h>

class Swarm {
    public:
        Swarm(int num_drones, int K, int n, float delta_t, Eigen::VectorXd p_min, Eigen::VectorXd p_max, float w_g_p, float w_g_v, float w_s, int kappa, float v_bar, float f_bar, std::vector<Eigen::VectorXd> initial_positions, std::vector<Eigen::MatrixXd> waypoints, std::string& params_filepath);
        Swarm(); // default constructor - this should be removed later, quick hack to allow simulator to keep a swarm as a member variable on initialization

        void solve(const double);
        int num_drones;
        int K;
        std::vector<Drone> drones;
        std::vector<Eigen::SparseMatrix<double>> all_thetas;

        struct CollisionParameters {
            std::vector<Eigen::SparseMatrix<double>> thetas;
            Eigen::VectorXd xi;
        };

        
};


#endif