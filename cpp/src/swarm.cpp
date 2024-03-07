#include <swarm.h>
#include <utils.h>
#include <stdexcept>


using namespace Eigen;


Swarm::Swarm(std::vector<std::shared_ptr<Drone>> drones)
    : drones(std::move(drones))
{   
    num_drones = this->drones.size();
    if (!this->drones.empty()) {
        int K = this->drones[0]->getK();
        // create collision matrices
        SparseMatrix<double> eyeKp1 = SparseMatrix<double>(K+1,K+1);
        eyeKp1.setIdentity();
        for (int i = 0; i < num_drones; ++i) {
            // create vector defining collision envelope for each drone across all time steps
            // at each time step, each drone will take the relevant thetas from this vector
            all_obstacle_envelopes.push_back(utils::kroneckerProduct(eyeKp1, this->drones[i]->getCollisionEnvelope())); // contains collision envelope for all drones for all time steps
        }
    }
};


std::pair<std::vector<bool>,std::vector<DroneResult>> Swarm::solve(double current_time,
                                                                    const std::vector<VectorXd>& initial_states,
                                                                    const std::vector<DroneResult>& previous_results,
                                                                    const std::vector<ConstraintConfig>& constraint_configs) {
    if (initial_states.size() != num_drones ||
        previous_results.size() != num_drones ||
        constraint_configs.size() != num_drones) {
        throw std::invalid_argument("Input vectors must all have the same length as the number of drones in the swarm.");
    }

    int K = drones[0]->getK();

    // initialize results
    std::vector<bool> is_success(num_drones);
    std::vector<DroneResult> results(num_drones);
    
    # pragma omp parallel for
    for (int i = 0; i < drones.size(); ++i) {
        std::vector<VectorXd> obstacle_positions;
        std::vector<SparseMatrix<double>> obstacle_envelopes;
        int num_obstacles = 0;

        for (int drone = 0; drone < drones.size(); ++drone) {
            if (drone < i && Swarm::checkIntersection(previous_results[i].position_trajectory_vector, previous_results[drone].position_trajectory_vector, 0.75*all_obstacle_envelopes[drone])) { // TODO change magic number
                obstacle_positions.push_back(previous_results[drone].position_trajectory_vector);
                obstacle_envelopes.push_back(all_obstacle_envelopes[drone]);
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

        std::pair<bool, DroneResult> result = drones[i]->solve(args);
        
        // use a critical section to update shared vectors
        # pragma omp critical
        {
            is_success[i] = result.first;
            results[i] = result.second;
        }
    }
    
    return {is_success, results};
}

bool Swarm::checkIntersection(const VectorXd& traj1, const VectorXd& traj2, const SparseMatrix<double>& theta) {
    VectorXd diff = theta*(traj1 - traj2);
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