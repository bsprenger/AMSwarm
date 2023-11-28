#include <swarm.h>
#include <utils.h>
#include <iostream>
#include <thread>


Swarm::Swarm() {
    // Default initialization logic
}

Swarm::Swarm(int num_drones, int K, int n, float delta_t, Eigen::VectorXd p_min, Eigen::VectorXd p_max, float w_g_p, float w_g_v, float w_s, int kappa, float v_bar, float f_bar, std::unordered_map<int, Eigen::VectorXd> initial_positions, std::unordered_map<int, Eigen::MatrixXd> waypoints, std::string& params_filepath)
    : num_drones(num_drones), K(K)
{
    
    // set size of drone vector
    drones.reserve(num_drones);

    // hack to get drone IDs for now
    std::vector<int> drone_ids;
    for(std::unordered_map<int, Eigen::VectorXd>::iterator it = initial_positions.begin(); it != initial_positions.end(); ++it) {
        drone_ids.push_back(it->first);
    }
    // create drones
    for (int i = 0; i < num_drones; ++i) {
        drones.emplace_back(Drone(params_filepath, waypoints[drone_ids[i]], initial_positions[drone_ids[i]], K, n, delta_t, p_min, p_max, w_g_p, w_g_v, w_s, v_bar, f_bar));
        
        // create vector defining collision envelope for each drone across all time steps
        // at each time step, each drone will take the relevant thetas from this vector
        Eigen::SparseMatrix<double> eyeK = Eigen::SparseMatrix<double>(K,K);
        eyeK.setIdentity();
        all_thetas.push_back(utils::kroneckerProduct(eyeK, drones[i].collision_envelope)); // contains collision envelope for all drones for all time steps
    }
};


void Swarm::solve(const double current_time) {
    // Vector to hold threads
    std::vector<std::thread> threads;

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
                xi.segment(3 * K * index, 3 * K) = drones[drone].pos_traj_vector;
                ++index;
            }
        }

        // add collision parameters to vector
        collisionParameters.push_back({thetas, xi});
    }

    // now we have a vector of drones and a corresponding vector of collisionParameters
    // we can now solve each drone's trajectory in parallel
    // to do: multithread this
    for (int i = 0; i < drones.size(); ++i) {
        drones[i].solve(current_time, drones[i].state_traj_vector.head(6), j, collisionParameters[i].thetas, collisionParameters[i].xi);
    }
}