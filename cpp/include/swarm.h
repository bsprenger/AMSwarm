#ifndef SWARM_H
#define SWARM_H

#include <drone.h>


using namespace Eigen;


class Swarm {
    public:
        class SwarmResult {
            public:
                std::vector<Drone::DroneResult> drone_results;

                // accessor for indidivual drone data
                const Drone::DroneResult& getDroneResult(int drone_index) const {
                    return drone_results.at(drone_index);
                }
        };

        Swarm(std::vector<Drone> drones);

        SwarmResult solve(const double current_time,
                            std::vector<VectorXd> x_0_vector, // rename these
                            std::vector<VectorXd> prev_trajectories,
                            std::vector<Drone::SolveOptions> opt,
                            std::vector<VectorXd> prev_inputs = std::vector<VectorXd>(0));

        SwarmResult runSimulation();

    private:
        int num_drones;
        std::vector<Drone> drones;
        std::vector<SparseMatrix<double>> all_thetas; // to do more elegant solution 
};

#endif