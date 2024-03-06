#ifndef SWARM_H
#define SWARM_H

#include <drone.h>


using namespace Eigen;


class Swarm {
    public:
        Swarm(std::vector<std::shared_ptr<Drone>> drones);

        std::pair<std::vector<bool>, std::vector<DroneResult>> solve(const double current_time,
                            std::vector<VectorXd> x_0_vector, // rename these
                            std::vector<VectorXd> prev_trajectories,
                            std::vector<VectorXd> prev_inputs = std::vector<VectorXd>());

    private:
        int num_drones;
        std::vector<std::shared_ptr<Drone>> drones;
        std::vector<SparseMatrix<double>> all_obstacle_envelopes; // to do more elegant solution 

        bool checkIntersection(const VectorXd& traj1, const VectorXd& traj2, const SparseMatrix<double>& theta_tmp);
};

#endif