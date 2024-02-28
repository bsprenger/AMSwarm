#include <swarm.h>
#include <utils.h>
#include <iostream>


using namespace Eigen;


Swarm::Swarm(std::vector<Drone> drones)
    : drones(drones)
{
    // get length of drones vector
    num_drones = drones.size();
    int K = drones[0].getK();

    // create collision matrices
    for (int i = 0; i < num_drones; ++i) {
        // create vector defining collision envelope for each drone across all time steps
        // at each time step, each drone will take the relevant thetas from this vector
        SparseMatrix<double> eyeKp1 = SparseMatrix<double>(K+1,K+1);
        eyeKp1.setIdentity();
        all_thetas.push_back(utils::kroneckerProduct(eyeKp1, drones[i].getCollisionEnvelope())); // contains collision envelope for all drones for all time steps
    }
};


Swarm::SwarmResult Swarm::solve(const double current_time,
                                std::vector<VectorXd> x_0_vector,
                                std::vector<VectorXd> prev_trajectories,
                                std::vector<Drone::SolveOptions> opt,
                                std::vector<VectorXd> prev_inputs) {
    int K = drones[0].getK();
    // int j = num_drones - 1; // for now, consider all drones as obstacles - later only consider some within radius

    SwarmResult swarm_result;
    swarm_result.drone_results.resize(num_drones);

    SparseMatrix<double> eyeKp1 = SparseMatrix<double>(K+1,K+1);
    eyeKp1.setIdentity();
    SparseMatrix<double> buffer(3,3);
    buffer.insert(0,0) = 5; buffer.insert(1,1) = 5; buffer.insert(2,2) = 5;
    SparseMatrix<double> theta_intersection_buffer = utils::kroneckerProduct(eyeKp1, buffer);
    
    # pragma omp parallel for
    for (int i = 0; i < drones.size(); ++i) {
        // std::vector<SparseMatrix<double>> thetas = all_thetas;
        // thetas.erase(thetas.begin() + i);
        // VectorXd xi(3 * (K+1) * j);
        Eigen::VectorXd xi;
        std::vector<SparseMatrix<double>> thetas;
        std::vector<int> intersecting_drones;
        // assign each each obstacle's trajectory to xi - in this case, all OTHER drones
        for (int drone = 0; drone < drones.size(); ++drone) {
            if (drone != i && drone < i && Swarm::checkIntersection(prev_trajectories[i], prev_trajectories[drone], all_thetas[drone] + theta_intersection_buffer)) {
                // If there is an intersection, append this drone's trajectory to xi
                VectorXd to_append = prev_trajectories[drone];
                if (xi.size() == 0) {
                    xi = to_append;
                } else {
                    xi.conservativeResize(xi.size() + to_append.size());
                    xi.tail(to_append.size()) = to_append;
                }

                // Also append the corresponding theta to thetas
                thetas.push_back(all_thetas[drone]);
                intersecting_drones.push_back(drone); // here
            }
        }
        
        // At this point, xi only contains trajectories of intersecting drones and thetas their thetas
        int j = thetas.size(); // j is now the number of intersecting drones
        // std::cout << "j Size" << j << std::endl;
        // std::cout << "Xi Size" << xi.size() << std::endl;
        // std::cout << "thetas Size" << thetas.size() << std::endl;
        // std::cout << "ID: " << i << std::endl;
        // std::cout << "x_0_vector" << prev_inputs[i] << std::endl;
        
        Drone::DroneResult result;
        if (!prev_inputs.empty()) {
            result = drones[i].solve(current_time, x_0_vector[i], j, thetas, xi, opt[i],
                                    prev_inputs[i]);
        } else {
            result = drones[i].solve(current_time, x_0_vector[i], j, thetas, xi, opt[i]);
        }
        // VectorXd initial_guess_control_input_trajectory_vector = prev_inputs[i];
        // Drone::DroneResult result = drones[i].solve(current_time, x_0_vector[i],
        //                                             j, thetas, xi, opt[i],
        //                                             initial_guess_control_input_trajectory_vector);
        

        // use a critical section to update shared vectors
        # pragma omp critical
        {
            swarm_result.drone_results[i] = result;
        }
    }
    
    return swarm_result;
}

bool Swarm::checkIntersection(const VectorXd& traj1, const VectorXd& traj2, const SparseMatrix<double>& theta) {
    VectorXd diff = theta*(traj1 - traj2);
    // Iterate over chunks of 3 rows
    for (int i = 0; i < diff.size(); i += 3) {
        // Ensure not to exceed the vector's bounds
        // int chunkSize = std::min(3, int(diff.size() - i));
        // VectorXd chunkDiff = diff.segment(i, chunkSize);
        // double norm = (theta_tmp.block(i, 0, chunkSize, diff.size()) * chunkDiff).norm();
        double norm = diff.segment(i, 3).norm();
        // If the norm of any chunk is less than or equal to 1, return true for an intersection
        if (norm <= 1.0) {
            return true;
        }
    }
    // If no chunk's norm was <= 1, then no intersection was detected
    return false;
}


// Swarm::SwarmResult Swarm::runSimulation() {
//     int K = drones[0].getK();

//     float delta_t = drones[0].getDeltaT();
    
//     SwarmResult swarm_result;
//     swarm_result.drone_results.resize(num_drones);

//     for (int i = 0; i < num_drones; ++i) {
//         VectorXd row(3);
//         row << drones[i].getInitialPosition();
//         swarm_result.drone_results[i].position_trajectory.conservativeResize(1, 3);
//         swarm_result.drone_results[i].position_trajectory.row(0) = row;
//         swarm_result.drone_results[i].position_state_time_stamps.conservativeResize(1);
//         swarm_result.drone_results[i].position_state_time_stamps(0) = 0.0;
//     }

//     // get max waypoint time (iterate over each drone, and get max time from waypoints)
//     double final_waypoint_time = 0.0;
//     for (int i = 0; i < num_drones; ++i) {
//         double max_time = drones[i].getWaypoints().col(0).maxCoeff();
//         if (max_time > final_waypoint_time) {
//             final_waypoint_time = max_time;
//         }
//     }

//     // round final_waypoint_time to nearest delta_t
//     final_waypoint_time = std::round(final_waypoint_time / delta_t) * delta_t;
    
//     // initialize trajectory vectors to be the initial positions
//     std::vector<VectorXd> prev_inputs;
//     std::vector<VectorXd> prev_trajectories;
//     std::vector<VectorXd> x_0_vector;
//     for (int i = 0; i < num_drones; ++i) {
//         prev_inputs.push_back(drones[i].getInitialPosition().replicate(K,1));
//         prev_trajectories.push_back(drones[i].getInitialPosition().replicate(K,1));
//         VectorXd initial_state(6); initial_state << drones[i].getInitialPosition(), VectorXd::Zero(3);
//         x_0_vector.push_back(initial_state);
//     }

//     // solve for initial trajectories THIS CAN BE IMPROVED
//     // create a vector of bools of all false for the drones
//     std::vector<Drone::SolveOptions> opt;
//     SwarmResult solve_result = solve(0.0, x_0_vector, prev_trajectories,
//                                     opt, prev_inputs);
//     prev_inputs.clear();
//     prev_trajectories.clear();
//     for (int i = 0; i < num_drones; ++i) {
//         prev_inputs.push_back(solve_result.drone_results[i].control_input_trajectory_vector);
//         prev_trajectories.push_back(solve_result.drone_results[i].position_trajectory_vector);
//     }

//     // iterate over time
//     for (float t = 0.0; t < final_waypoint_time - delta_t; t+=delta_t) {
//         // Solve for next trajectories
//         solve_result = solve(t, x_0_vector, prev_trajectories,
//                             opt, prev_inputs);

//         // Build necessary matrices
//         // previous trajectories need to be moved one time step forward and the last time step needs to be extrapolated
//         prev_inputs.clear();
//         prev_trajectories.clear();
//         x_0_vector.clear();
//         // Get simulation output - positions and control inputs
//         for (int j = 0; j < num_drones; ++j) {
//             prev_inputs.push_back(solve_result.drone_results[j].control_input_trajectory_vector);
//             prev_trajectories.push_back(solve_result.drone_results[j].position_trajectory_vector);
//             x_0_vector.push_back(solve_result.drone_results[j].state_trajectory_vector.head(6)); // THIS NEEDS TO BE FIXED, FIRST ITERATION SHOULD NOT HAVE VELOCITY

//             // time
//             swarm_result.drone_results[j].control_input_time_stamps.conservativeResize(swarm_result.drone_results[j].control_input_time_stamps.size() + 1);
//             swarm_result.drone_results[j].control_input_time_stamps(swarm_result.drone_results[j].control_input_time_stamps.size() - 1) = t;

//             swarm_result.drone_results[j].position_state_time_stamps.conservativeResize(swarm_result.drone_results[j].position_state_time_stamps.size() + 1);
//             swarm_result.drone_results[j].position_state_time_stamps(swarm_result.drone_results[j].position_state_time_stamps.size() - 1) = t + delta_t;

//             // First, inputs at current time
//             VectorXd input_row(3); // time, x, y, z
//             input_row << solve_result.drone_results[j].control_input_trajectory_vector[0], solve_result.drone_results[j].control_input_trajectory_vector[1], solve_result.drone_results[j].control_input_trajectory_vector[2];
//             int current_input_size = swarm_result.drone_results[j].control_input_trajectory.rows();
//             swarm_result.drone_results[j].control_input_trajectory.conservativeResize(current_input_size + 1, 3);
//             swarm_result.drone_results[j].control_input_trajectory.row(current_input_size) = input_row;

//             // Second, predicted positions at the next time
//             VectorXd pos_row(3); // time, x, y, z
//             pos_row << solve_result.drone_results[j].position_trajectory_vector[0], solve_result.drone_results[j].position_trajectory_vector[1], solve_result.drone_results[j].position_trajectory_vector[2]; // next position predicted, not current position
//             int current_position_size = swarm_result.drone_results[j].position_trajectory.rows();
//             swarm_result.drone_results[j].position_trajectory.conservativeResize(current_position_size + 1, 3);
//             swarm_result.drone_results[j].position_trajectory.row(current_position_size) = pos_row;
//         }
        
//     }
//     return swarm_result;
// }