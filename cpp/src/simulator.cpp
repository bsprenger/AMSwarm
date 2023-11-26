#include <simulator.h>
#include <iostream>
#include <fstream>

Simulator::Simulator(int num_drones, int K, int n, float delta_t, Eigen::VectorXd p_min, Eigen::VectorXd p_max,
                    float w_g_p, float w_g_v, float w_s, int kappa, float v_bar, float f_bar, std::unordered_map<int, Eigen::VectorXd> initial_positions, std::unordered_map<int, Eigen::MatrixXd> waypoints, std::string& params_filepath) : 
                        num_drones(num_drones), K(K), n(n), delta_t(delta_t), p_min(p_min), p_max(p_max), w_g_p(w_g_p),
                        w_g_v(w_g_v), w_s(w_s), kappa(kappa), v_bar(v_bar), f_bar(f_bar), initial_positions(initial_positions), waypoints(waypoints) {

    swarm = Swarm(num_drones, K, n, delta_t, p_min, p_max, w_g_p, w_g_v, w_s, kappa, v_bar, f_bar, initial_positions, waypoints, params_filepath);
    
};

std::unordered_map<int, Eigen::MatrixXd> Simulator::runSimulation() {
    float t = 0.0;

    // get max waypoint time (1st column of each MatrixXd from waypoints unordered map)
    double final_waypoint_time = 0.0;
    for (const auto& item : waypoints) {
        Eigen::MatrixXd waypoint = item.second;
        double max_time = waypoint.col(0).maxCoeff();
        if (max_time > final_waypoint_time) {
            final_waypoint_time = max_time;
        }
    }

    final_waypoint_time = std::round(final_waypoint_time / delta_t) * delta_t; // round to nearest delta_t
    std::cout << final_waypoint_time << std::endl;
    // Eigen::MatrixXd result;
    std::unordered_map<int, Eigen::MatrixXd> positions;
    
    for (float t = 0.0; t < final_waypoint_time - delta_t; t+=delta_t) { // the -1e6 is to avoid floating point errors with frequencies that have irrational time stamps e.g. 48Hz, 6Hz
        for (int j = 0; j < num_drones; ++j) {
            Eigen::VectorXd row(4); // time, x, y, z
            row << t, swarm.drones[j].pos_traj_vector[0], swarm.drones[j].pos_traj_vector[1], swarm.drones[j].pos_traj_vector[2];

            positions[j].conservativeResize(positions[j].rows() + 1, 4);
            positions[j].row(positions[j].rows()-1) = row;
        }
        
        swarm.solve(t);
    }
    // get last position at final waypoint time
    for (int j = 0; j < num_drones; ++j) {
        Eigen::VectorXd row(4); // time, x, y, z
        row << final_waypoint_time, swarm.drones[j].pos_traj_vector[0], swarm.drones[j].pos_traj_vector[1], swarm.drones[j].pos_traj_vector[2];
        
        positions[j].conservativeResize(positions[j].rows() + 1, 4);
        positions[j].row(positions[j].rows()-1) = row;
    }
    return positions;
}