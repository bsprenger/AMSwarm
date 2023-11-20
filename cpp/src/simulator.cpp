#include <simulator.h>
#include <iostream>
// #include <yaml-cpp/yaml.h>
#include <fstream>

Simulator::Simulator(const Eigen::MatrixXd& waypoints) {
    std::cout << "Created simulator with waypoints:" << std::endl;
    std::cout << waypoints << std::endl;
};

Eigen::MatrixXd Simulator::runSimulation() {
    Eigen::MatrixXd result = Eigen::MatrixXd::Random(3, 3);

    return result;
}