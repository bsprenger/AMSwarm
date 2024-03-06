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
                                                                    bool is_initial_solve) {
    if (initial_states.size() != num_drones ||
        previous_results.size() != num_drones) {
        throw std::invalid_argument("Input vectors must all have the same length as the number of drones in the swarm.");
    }

    int K = drones[0]->getK();

    // initialize results
    std::vector<bool> is_success(num_drones);
    std::vector<DroneResult> results(num_drones);

    SparseMatrix<double> eyeKp1 = SparseMatrix<double>(K+1,K+1);
    eyeKp1.setIdentity();
    SparseMatrix<double> buffer(3,3);
    buffer.insert(0,0) = -0.5; buffer.insert(1,1) = -0.5; buffer.insert(2,2) = -0.5;
    SparseMatrix<double> theta_intersection_buffer = utils::kroneckerProduct(eyeKp1, buffer);
    
    # pragma omp parallel for
    for (int i = 0; i < drones.size(); ++i) {
        VectorXd drone_trajectory;
        if (is_initial_solve) {
            drone_trajectory = previous_results[i].position_trajectory_vector;
        } else {
            drone_trajectory = Drone::updateAndExtrapolateTrajectory(previous_results[i].position_trajectory_vector);
        }

        std::vector<VectorXd> obstacle_positions;
        std::vector<SparseMatrix<double>> obstacle_envelopes;
        int num_obstacles = 0;

        for (int drone = 0; drone < drones.size(); ++drone) {
            VectorXd other_drone_trajectory;
            if (is_initial_solve) {
                other_drone_trajectory = previous_results[drone].position_trajectory_vector;
            } else {
                other_drone_trajectory = Drone::updateAndExtrapolateTrajectory(previous_results[drone].position_trajectory_vector);
            }
            if (drone < i && Swarm::checkIntersection(drone_trajectory, other_drone_trajectory, all_obstacle_envelopes[drone] + theta_intersection_buffer)) {
                obstacle_positions.push_back(other_drone_trajectory);
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
        int row_index = is_initial_solve ? 0 : 1; // if it's the initial solve, use the first row of the trajectory i.e. do not advance the guess, otherwise use the second row
        args.u_0 = previous_results[i].input_position_trajectory.row(row_index);
        args.u_dot_0 = previous_results[i].input_velocity_trajectory.row(row_index);
        args.u_ddot_0 = previous_results[i].input_acceleration_trajectory.row(row_index);

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