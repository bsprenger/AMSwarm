#include <simulator.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>

Simulator::Simulator(const std::string& yamlFile) {
    // Load YAML file
    YAML::Node config = YAML::LoadFile(yamlFile);
    num_drones = config["num_drones"].as<int>();

    const YAML::Node& positionsNode = config["initial_positions"];
    for (const auto& position : positionsNode) {
        int index = position[0].as<int>();
        Eigen::VectorXd vec(3);
        vec << position[1].as<double>(), position[2].as<double>(), position[3].as<double>();
        initial_positions.resize(std::max(index + 1, static_cast<int>(initial_positions.size())), Eigen::VectorXd(3));
        initial_positions[index] = vec;
    }

    // Assign waypoints
    const YAML::Node& waypointsNode = config["waypoints"];
    for (const auto& waypoint : waypointsNode) {
        int index = waypoint[0].as<int>();
        Eigen::VectorXd row(7);
        row << waypoint[1].as<double>(), waypoint[2].as<double>(), waypoint[3].as<double>(),
               waypoint[4].as<double>(), waypoint[5].as<double>(), waypoint[6].as<double>(),
               waypoint[7].as<double>();
        waypoints.resize(std::max(index + 1, static_cast<int>(waypoints.size())), Eigen::MatrixXd(0, 7));
        waypoints[index].conservativeResize(waypoints[index].rows() + 1, Eigen::NoChange);
        waypoints[index].row(waypoints[index].rows() - 1) = row;
    }

    K = config["K"].as<int>();
    n = config["n"].as<int>();
    delta_t = config["delta_t"].as<float>();
    // Assign p_min
    const YAML::Node& pMinNode = config["p_min"];
    p_min.resize(pMinNode.size());
    for (std::size_t i = 0; i < pMinNode.size(); ++i) {
        p_min[i] = pMinNode[i].as<double>();
    }
    // Assign p_max
    const YAML::Node& pMaxNode = config["p_max"];
    p_max.resize(pMaxNode.size());
    for (std::size_t i = 0; i < pMaxNode.size(); ++i) {
        p_max[i] = pMaxNode[i].as<double>();
    }
    w_g_p = config["w_g_p"].as<float>();
    w_g_v = config["w_g_v"].as<float>();
    w_s = config["w_s"].as<float>();
    kappa = config["kappa"].as<int>();
    v_bar = config["v_bar"].as<float>();
    f_bar = config["f_bar"].as<float>();

    // std::cout << initial_positions[0] << std::endl;
    swarm = Swarm(num_drones, K, n, delta_t, p_min, p_max, w_g_p, w_g_v, w_s, kappa, v_bar, f_bar, initial_positions, waypoints);
};

void Simulator::run() {
    double t = 0.0;
    std::ofstream outputFile("output.csv");
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open output file!" << std::endl;
        return;
    }
    for (int i = 0;i < 25;++i) {
        swarm.solve(t);
        t += delta_t;

        for (int j = 0; j < num_drones; ++j) {
            // Write data to the CSV file
            outputFile << j << "," << t << ","
                        << swarm.drones[j].pos_traj_vector[0] << ","
                        << swarm.drones[j].pos_traj_vector[1] << ","
                        << swarm.drones[j].pos_traj_vector[2] << std::endl;
        }

        std::cout << "t = " << t << std::endl;
        for (int j = 0;j < num_drones;++j) {
            std::cout << "Drone " << j << " position: " << swarm.drones[j].pos_traj_vector.head(3) << std::endl;
        }
    }
    outputFile.close();
};