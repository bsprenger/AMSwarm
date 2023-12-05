#include <swarm.h>
#include <utils.h>
#include <iostream>


Swarm::Swarm() {
    // Default initialization logic
}

Swarm::Swarm(std::vector<Drone> drones, int K)
    : drones(drones), K(K)
{
    // get length of drones vector
    num_drones = drones.size();

    // create collision matrices
    for (int i = 0; i < num_drones; ++i) {
        // create vector defining collision envelope for each drone across all time steps
        // at each time step, each drone will take the relevant thetas from this vector
        Eigen::SparseMatrix<double> eyeK = Eigen::SparseMatrix<double>(K,K);
        eyeK.setIdentity();
        all_thetas.push_back(utils::kroneckerProduct(eyeK, drones[i].getCollisionEnvelope())); // contains collision envelope for all drones for all time steps

        // initialize trajectories vector, assuming that each drone stays in its initial position over the horizon. this will be updated after each optimization routine
        pos_trajectories.push_back(drones[i].getInitialPosition().replicate(K,1));
        Eigen::VectorXd initial_state(6); initial_state << drones[i].getInitialPosition(), Eigen::VectorXd::Zero(3);
        state_trajectories.push_back(initial_state.replicate(K,1));
    }
};


std::vector<Drone::OptimizationResult> Swarm::solve(const double current_time) {
    // Vector to hold collision variables for each drone
    std::vector<CollisionParameters> collisionParameters;

    // Get number of obstacles, both dynamic and static
    int j = num_drones - 1; // for now, consider all drones as obstacles - later only consider some within radius
    
    // Get obstacle collision envelopes and paths for each drone
    for (int i = 0; i < drones.size(); ++i) {
        // Get all the collision envelopes for all OTHER drones - start will all envelopes including for self, then remove theta for self
        // this should be changed at some point to only select relevant obstacles within radius
        std::vector<Eigen::SparseMatrix<double>> thetas = all_thetas;
        thetas.erase(thetas.begin() + i); // remove the collision envelope for drone i - drone should not consider "self collisions"

        // initialize empty vector to store paths of all obstacles
        Eigen::VectorXd xi(3 * K * j);

        // assign each each obstacle's trajectory to xi - in this case, all OTHER drones
        int index = 0;
        for (int drone = 0; drone < drones.size(); ++drone) {
            if (drone != i) {
                // xi.segment(3 * K * index, 3 * K) = drones[drone].pos_traj_vector;
                xi.segment(3 * K * index, 3 * K) = pos_trajectories[drone];
                ++index;
            }
        }

        // add collision parameters to vector
        collisionParameters.push_back({thetas, xi});
    }

    // now we have a vector of drones and a corresponding vector of collisionParameters
    // we can now solve each drone's trajectory in parallel
    std::vector<Drone::OptimizationResult> results(drones.size());

    # pragma omp parallel for
    for (int i = 0; i < drones.size(); ++i) {
        Drone::OptimizationResult result = drones[i].solve(current_time, state_trajectories[i].head(6), j, collisionParameters[i].thetas, collisionParameters[i].xi);

        // use a critical section to update shared vectors
        # pragma omp critical
        {
            results[i] = result;;
            pos_trajectories[i] = result.pos_traj_vector;
            state_trajectories[i] = result.state_traj_vector;
        }
    }
    return results;
}