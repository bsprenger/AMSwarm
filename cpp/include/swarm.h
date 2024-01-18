#ifndef SWARM_H
#define SWARM_H

#include <drone.h>

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
                            std::vector<Eigen::VectorXd> x_0_vector, // rename these
                            std::vector<Eigen::VectorXd> prev_inputs,
                            std::vector<Eigen::VectorXd> prev_trajectories,
                            std::vector<bool> waypoint_position_constraints,
                            std::vector<bool> waypoint_velocity_constraints,
                            std::vector<bool> waypoint_acceleration_constraints,
                            std::vector<bool> input_continuity_constraints,
                            std::vector<bool> input_dot_continuity_constraints,
                            std::vector<bool> input_ddot_continuity_constraints);

        SwarmResult runSimulation();

    private:
        int num_drones;
        std::vector<Drone> drones;
        std::vector<Eigen::SparseMatrix<double>> all_thetas; // to do more elegant solution 
};

#endif