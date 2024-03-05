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
        all_obstacle_envelopes.push_back(utils::kroneckerProduct(eyeKp1, drones[i].getCollisionEnvelope())); // contains collision envelope for all drones for all time steps
    }
};


std::pair<std::vector<bool>,std::vector<DroneResult>> Swarm::solve(const double current_time,
                                                                            std::vector<VectorXd> x_0_vector,
                                                                            std::vector<VectorXd> prev_trajectories,
                                                                            std::vector<VectorXd> prev_inputs) {
    int K = drones[0].getK();

    // initialize results
    std::vector<bool> is_success(num_drones);
    std::vector<DroneResult> results(num_drones);

    SparseMatrix<double> eyeKp1 = SparseMatrix<double>(K+1,K+1);
    eyeKp1.setIdentity();
    SparseMatrix<double> buffer(3,3);
    buffer.insert(0,0) = 5; buffer.insert(1,1) = 5; buffer.insert(2,2) = 5;
    SparseMatrix<double> theta_intersection_buffer = utils::kroneckerProduct(eyeKp1, buffer);
    
    # pragma omp parallel for
    for (int i = 0; i < drones.size(); ++i) {
        std::vector<VectorXd> obstacle_positions;
        std::vector<SparseMatrix<double>> obstacle_envelopes;
        int num_obstacles = 0;

        for (int drone = 0; drone < drones.size(); ++drone) {
            if (drone < i && Swarm::checkIntersection(prev_trajectories[i], prev_trajectories[drone], all_obstacle_envelopes[drone] + theta_intersection_buffer)) {
                obstacle_positions.push_back(prev_trajectories[drone]);
                obstacle_envelopes.push_back(all_obstacle_envelopes[drone]);
                num_obstacles++;
            }
        }

        DroneSolveArgs args;
        args.current_time = current_time;
        args.num_obstacles = num_obstacles;
        args.obstacle_positions = obstacle_positions;
        args.obstacle_envelopes = obstacle_envelopes;
        args.x_0 = x_0_vector[i];
        
        std::pair<bool, DroneResult> result = drones[i].solve(args);
        
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