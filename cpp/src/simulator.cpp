#include <simulator.h>
#include <iostream>
#include <fstream>

Simulator::Simulator(int num_drones, int K, int n, float delta_t, Eigen::VectorXd p_min, Eigen::VectorXd p_max,
                    float w_g_p, float w_g_v, float w_s, int kappa, float v_bar, float f_bar, std::map<int, Eigen::VectorXd> initial_positions, std::map<int, Eigen::MatrixXd> waypoints, std::string& params_filepath) : 
                        num_drones(num_drones), K(K), n(n), delta_t(delta_t), p_min(p_min), p_max(p_max), w_g_p(w_g_p),
                        w_g_v(w_g_v), w_s(w_s), kappa(kappa), v_bar(v_bar), f_bar(f_bar), initial_positions(initial_positions), waypoints(waypoints) {

    std::vector<Drone> drones;
    drones.reserve(num_drones);

    // hack to get drone IDs for now
    std::vector<int> drone_ids;
    for(std::map<int, Eigen::VectorXd>::iterator it = initial_positions.begin(); it != initial_positions.end(); ++it) {
        drone_ids.push_back(it->first);
    }

    // create drones
    for (int i = 0; i < num_drones; ++i) {
        drones.emplace_back(Drone(params_filepath, waypoints[drone_ids[i]], initial_positions[drone_ids[i]], K, n, delta_t, p_min, p_max, w_g_p, w_g_v, w_s, v_bar, f_bar));
    }

    // initialize swarm
    swarm = Swarm(drones,K);
    
};

std::map<int, Eigen::MatrixXd> Simulator::runSimulation() {
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
    // Eigen::MatrixXd result;
    std::map<int, Eigen::MatrixXd> positions;

    std::vector<Drone::OptimizationResult> result = swarm.solve(0.0); // todo: initial optimization routine for first position when they don't have each others' trajectories
    
    for (float t = 0.0; t < final_waypoint_time - delta_t; t+=delta_t) { // todo something more elegant than this
        for (int j = 0; j < num_drones; ++j) {
            Eigen::VectorXd row(4); // time, x, y, z
            row << t, result[j].input_traj_vector[0], result[j].input_traj_vector[1], result[j].input_traj_vector[2];

            positions[j].conservativeResize(positions[j].rows() + 1, 4);
            positions[j].row(positions[j].rows()-1) = row;
        }
        
        result = swarm.solve(t);
    }
    // get last position at final waypoint time
    for (int j = 0; j < num_drones; ++j) {
        Eigen::VectorXd row(4); // time, x, y, z
        row << final_waypoint_time, result[j].input_traj_vector[0], result[j].input_traj_vector[1], result[j].input_traj_vector[2];
        
        positions[j].conservativeResize(positions[j].rows() + 1, 4);
        positions[j].row(positions[j].rows()-1) = row;
    }
    return positions;
}