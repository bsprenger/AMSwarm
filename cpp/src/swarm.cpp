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