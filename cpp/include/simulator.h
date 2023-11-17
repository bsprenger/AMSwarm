#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <swarm.h>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

class Simulator {
    public:
        Simulator(const std::string&);
        void run();

        int num_drones;
        std::vector<Eigen::VectorXd> initial_positions;
        std::vector<Eigen::MatrixXd> waypoints;
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