#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <swarm.h>
#include <string>
#include <vector>
#include <Eigen/Dense>
// #include <yaml-cpp/yaml.h>

class Simulator {
    public:
        Simulator(int, int, int, float, Eigen::VectorXd, Eigen::VectorXd, float, float, float, int, float, float, std::unordered_map<int, Eigen::VectorXd>, std::unordered_map<int, Eigen::MatrixXd>, std::string&);

        std::unordered_map<int, Eigen::MatrixXd> runSimulation();

        int num_drones;
        std::unordered_map<int, Eigen::MatrixXd> waypoints;
        std::unordered_map<int, Eigen::VectorXd> initial_positions;
        int K;
        int n;
        float delta_t;
        Eigen::VectorXd p_min;
        Eigen::VectorXd p_max;
        float w_g_p;
        float w_g_v;
        float w_s;
        int kappa;
        float v_bar;
        float f_bar;

        Swarm swarm;

};



#endif