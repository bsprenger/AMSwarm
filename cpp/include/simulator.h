#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <swarm.h>
#include <string>
#include <vector>
#include <Eigen/Dense>
// #include <yaml-cpp/yaml.h>

class Simulator {
    public:
        Simulator(int, int, int, float, Eigen::VectorXd, Eigen::VectorXd, float, float, float, int, float, float, Eigen::MatrixXd, Eigen::MatrixXd, std::string&);

        Eigen::MatrixXd runSimulation();

        int num_drones;
        Eigen::MatrixXd waypoints;
        Eigen::MatrixXd initial_positions;
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