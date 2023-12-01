#ifndef SWARM_H
#define SWARM_H

#include <vector>
#include <drone.h>

class Swarm {
    public:
        Swarm(std::vector<Drone> drones, int K);
        Swarm(); // default constructor - this should be removed later, quick hack to allow simulator to keep a swarm as a member variable on initialization

        std::vector<Drone::OptimizationResult> solve(const double);
        int num_drones;
        int K;  // to do remove this
        std::vector<Drone> drones;
        std::vector<Eigen::SparseMatrix<double>> all_thetas;

        struct CollisionParameters {
            std::vector<Eigen::SparseMatrix<double>> thetas;
            Eigen::VectorXd xi;
        };

        std::vector<Eigen::VectorXd> pos_trajectories; // drone trajectories -> swarm has to keep track of each drone's trajectory so that it can be passed to the next optimization routine
        std::vector<Eigen::VectorXd> state_trajectories;
        
};


#endif