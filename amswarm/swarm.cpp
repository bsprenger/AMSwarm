#include "swarm.h"
#include "utils.h"
#include <stdexcept>

namespace amswarm {

using VectorXd = Eigen::VectorXd;
using SparseMatrixDouble = Eigen::SparseMatrix<double>;


Swarm::Swarm(std::vector<std::shared_ptr<Drone>> drones)
    : drones(std::move(drones))
{   
    num_drones = this->drones.size();
    if (!this->drones.empty()) {
        int K = this->drones[0]->getK(); // TODO: check if all drones have the same K
        
        // create collision matrices over all time steps for each drone
        SparseMatrixDouble eyeKp1 = SparseMatrixDouble(K+1,K+1); // TODO: replace with getSparseIdentity
        eyeKp1.setIdentity();
        
        // create a vector containing the collision envelopes for each drone across all time steps
        for (int i = 0; i < num_drones; ++i) {
            // at each time step, each drone will take the relevant collision envelopes from this vector according to which drones they need to avoid
            all_obstacle_envelopes.push_back(utils::kroneckerProduct(eyeKp1, this->drones[i]->getCollisionEnvelope()));
        }
    }
};


std::tuple<std::vector<bool>,std::vector<int>,std::vector<DroneResult>> Swarm::solve(double current_time,
                                                                    const std::vector<VectorXd>& initial_states,
                                                                    const std::vector<DroneResult>& previous_results,
                                                                    const std::vector<ConstraintConfig>& constraint_configs) {
    if (initial_states.size() != num_drones ||
        previous_results.size() != num_drones ||
        constraint_configs.size() != num_drones) {
        throw std::invalid_argument("Input vectors must all have the same length as the number of drones in the swarm.");
    }

    // Initialize avoidance responsibility counters for each drone
    std::vector<int> avoidance_counts(num_drones, 0);
    // Map each drone to a list of drones it should avoid
    std::unordered_map<int, std::vector<int>> avoidance_map;

    // Determine avoidance responsibilities
    for (int i = 0; i < num_drones; ++i) {
        for (int j = i + 1; j < num_drones; ++j) {
            // Decide which drone should avoid the other - see thesis document for details
            if (avoidance_counts[i] <= avoidance_counts[j]) {
                avoidance_map[i].push_back(j);
                avoidance_counts[i]++;
            } else {
                avoidance_map[j].push_back(i);
                avoidance_counts[j]++;
            }
        }
    }

    // initialize results
    std::vector<bool> is_success(num_drones);
    std::vector<int> iters(num_drones);
    std::vector<DroneResult> results(num_drones);
    
    // # pragma omp parallel for
    for (int i = 0; i < drones.size(); ++i) {
        std::vector<VectorXd> obstacle_positions;
        std::vector<SparseMatrixDouble> obstacle_envelopes;
        int num_obstacles = 0;

        for (const int& avoid_drone : avoidance_map[i]) {
            if (Swarm::checkIntersection(previous_results[i].position_trajectory_vector, previous_results[avoid_drone].position_trajectory_vector, 0.9*all_obstacle_envelopes[avoid_drone])) { // TODO change magic number
                obstacle_positions.push_back(previous_results[avoid_drone].position_trajectory_vector);
                obstacle_envelopes.push_back(all_obstacle_envelopes[avoid_drone]);
                num_obstacles++;
            }
        }

        DroneSolveArgs args;
        args.current_time = current_time;
        args.num_obstacles = num_obstacles;
        args.obstacle_positions = obstacle_positions;
        args.obstacle_envelopes = obstacle_envelopes;
        args.x_0 = initial_states[i];
        args.u_0 = previous_results[i].input_position_trajectory.row(0);
        args.u_dot_0 = previous_results[i].input_velocity_trajectory.row(0);
        args.u_ddot_0 = previous_results[i].input_acceleration_trajectory.row(0);
        args.constraintConfig = constraint_configs[i];

        std::tuple<bool, int, DroneResult> result = drones[i]->solve(args);
        
        // use a critical section to update shared vectors
        // # pragma omp critical
        // {
            is_success[i] = std::get<0>(result);
            iters[i] = std::get<1>(result);
            results[i] = std::get<2>(result);
        // }
    }
    
    return std::make_tuple(is_success, iters, results);
}

bool Swarm::checkIntersection(const VectorXd& traj1, const VectorXd& traj2, const SparseMatrixDouble& theta) {
    VectorXd diff = theta*(traj1 - traj2); // theta accounts for the collision envelopes by scaling the difference between the two trajectories
    // Iterate over chunks of 3 rows (x,y,z positions for one time)
    for (int i = 0; i < diff.size(); i += 3) {
        double norm = diff.segment(i, 3).norm();
        // If the norm of any chunk is less than or equal to 1, return true for an intersection
        if (norm <= 1.0) {
            return true;
        }
    }
    // If no chunk's norm was <= 1, then no intersection was detected
    return false;
}

} // namespace amswarm