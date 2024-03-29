#ifndef SWARM_H
#define SWARM_H

#include <drone.h>


using namespace Eigen;


class Swarm {
    public:
        Swarm(std::vector<std::shared_ptr<Drone>> drones);

        std::tuple<std::vector<bool>, std::vector<int>, std::vector<DroneResult>> solve(double current_time,
                            const std::vector<VectorXd>& initial_states,
                            const std::vector<DroneResult>& previous_results,
                            const std::vector<ConstraintConfig>& constraint_configs);

    private:
        int num_drones;
        std::vector<std::shared_ptr<Drone>> drones;
        std::vector<SparseMatrix<double>> all_obstacle_envelopes; // to do more elegant solution 
        // std::unordered_map<int, std::vector<int>> avoidance_map;

        bool checkIntersection(const VectorXd& traj1, const VectorXd& traj2, const SparseMatrix<double>& theta_tmp);
};

#endif