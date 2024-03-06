#ifndef SWARM_H
#define SWARM_H

#include <drone.h>


using namespace Eigen;


class Swarm {
    public:
        Swarm(std::vector<std::shared_ptr<Drone>> drones);

        std::pair<std::vector<bool>, std::vector<DroneResult>> solve(double current_time,
                            const std::vector<VectorXd>& initial_states,
                            const std::vector<DroneResult>& previous_results,
                            bool is_initial_solve);

    private:
        int num_drones;
        std::vector<std::shared_ptr<Drone>> drones;
        std::vector<SparseMatrix<double>> all_obstacle_envelopes; // to do more elegant solution 

        bool checkIntersection(const VectorXd& traj1, const VectorXd& traj2, const SparseMatrix<double>& theta_tmp);
};

#endif