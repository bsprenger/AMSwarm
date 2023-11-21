#include <simulator.h>
#include <iostream>
#include <fstream>

Simulator::Simulator(int num_drones, int K, int n, float delta_t, Eigen::VectorXd p_min, Eigen::VectorXd p_max,
                    float w_g_p, float w_g_v, float w_s, int kappa, float v_bar, float f_bar, Eigen::MatrixXd initial_positions, Eigen::MatrixXd waypoints, std::string& params_filepath) : 
                        num_drones(num_drones), K(K), n(n), delta_t(delta_t), p_min(p_min), p_max(p_max), w_g_p(w_g_p),
                        w_g_v(w_g_v), w_s(w_s), kappa(kappa), v_bar(v_bar), f_bar(f_bar), initial_positions(initial_positions), waypoints(waypoints) {
    
    // reformat initial_positions
    std::vector<Eigen::VectorXd> initial_positions_vector(num_drones);
    for (int i = 0; i < initial_positions.rows(); ++i) {
        // Extract the index from the first element of the row
        int index = static_cast<int>(initial_positions(i, 0));
        // Check if the index is within bounds of the vector
        if (index >= 0 && index < num_drones) {
            // Assign the remaining 6 elements of the row to the corresponding Eigen::VectorXd in the vector
            initial_positions_vector[index] = initial_positions.row(i).tail(3);
        } else {
            std::cerr << "Invalid drone ID in initial positions matrix: " << index << ". Check num_drones and check that 1st element in each row of initial_positions matrix is Drone ID in range 0 to num_drones-1." << std::endl;
        }
    }

    // reformat waypoints
    std::vector<Eigen::MatrixXd> waypoints_vector(num_drones);
    for (int i = 0; i < waypoints.rows(); ++i) {
        // Extract the index from the first element of the row
        int index = static_cast<int>(waypoints(i, 0));

        // Check if the index is within bounds of the vector
        if (index >= 0 && index < waypoints_vector.size()) {
            // Append the row (excluding the first element) to the corresponding Eigen::MatrixXd in the vector
            waypoints_vector[index].conservativeResize(waypoints_vector[index].rows() + 1, 7);
            waypoints_vector[index].row(waypoints_vector[index].rows() - 1) = waypoints.row(i).tail(7);
        } else {
            std::cerr << "Invalid index in waypoints matrix: " << index << ". Check num_drones and check that 1st element in each row of waypoints matrix is Drone ID in range 0 to num_drones-1." << std::endl;
            // Handle error as needed
        }
    }

    swarm = Swarm(num_drones, K, n, delta_t, p_min, p_max, w_g_p, w_g_v, w_s, kappa, v_bar, f_bar, initial_positions_vector, waypoints_vector, params_filepath);
};

Eigen::MatrixXd Simulator::runSimulation() {
    float t = 0.0;
    double final_waypoint_time = waypoints.col(1).maxCoeff();
    Eigen::MatrixXd result;
    
    for (float t = 0.0; t < final_waypoint_time; t+=delta_t) {
        for (int j = 0; j < num_drones; ++j) {
            Eigen::VectorXd row(5); // 1 extra element for the drone number
            row << j, t, swarm.drones[j].pos_traj_vector[0], swarm.drones[j].pos_traj_vector[1], swarm.drones[j].pos_traj_vector[2];
            
            result.conservativeResize(result.rows() + 1, 5);
            result.row(result.rows()-1) = row;
        }
        swarm.solve(t);
    }

    // get last position at final waypoint time
    for (int j = 0; j < num_drones; ++j) {
        Eigen::VectorXd row(5); // 1 extra element for the drone number
        row << j, final_waypoint_time, swarm.drones[j].pos_traj_vector[0], swarm.drones[j].pos_traj_vector[1], swarm.drones[j].pos_traj_vector[2];
        
        result.conservativeResize(result.rows() + 1, 5);
        result.row(result.rows()-1) = row;
    }

    return result;
}