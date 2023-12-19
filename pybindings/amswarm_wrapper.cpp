#include <iostream>
#include <swarm.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(amswarm, m)
{
    py::class_<Drone::DroneResult>(m, "DroneResult")
        .def(py::init<>())
        .def_readwrite("control_input_time_stamps", &Drone::DroneResult::control_input_time_stamps)
        .def_readwrite("position_state_time_stamps", &Drone::DroneResult::position_state_time_stamps)
        .def_readwrite("control_input_trajectory_vector", &Drone::DroneResult::control_input_trajectory_vector)
        .def_readwrite("state_trajectory_vector", &Drone::DroneResult::state_trajectory_vector)
        .def_readwrite("position_trajectory_vector", &Drone::DroneResult::position_trajectory_vector)
        .def_readwrite("control_input_trajectory", &Drone::DroneResult::control_input_trajectory)
        .def_readwrite("state_trajectory", &Drone::DroneResult::state_trajectory)
        .def_readwrite("position_trajectory", &Drone::DroneResult::position_trajectory);

    // Binding for SwarmResult
    py::class_<Swarm::SwarmResult>(m, "SwarmResult")
        .def(py::init<>())
        .def_readwrite("drone_results", &Swarm::SwarmResult::drone_results)
        .def("getDroneData", &Swarm::SwarmResult::getDroneResult);

    // Binding for DroneData
    // py::class_<Swarm::SwarmResult::DroneResult>(m, "DroneResult")
    //     .def(py::init<>())
    //     .def_readwrite("position_trajectory", &Swarm::SwarmResult::DroneResult::position_trajectory)
    //     .def_readwrite("control_input_trajectory", &Swarm::SwarmResult::DroneResult::control_input_trajectory);   

    py::class_<Drone>(m, "Drone")
        .def(py::init<std::string&, Eigen::MatrixXd, Eigen::VectorXd, bool, bool, int, int, float, Eigen::VectorXd, Eigen::VectorXd, float, float, float, float, float>())
        .def(py::init([](std::string& params_filepath,
                        py::array_t<double> waypoints_npy,
                        py::array_t<double> initial_pos_npy,
                        bool hard_waypoint_constraints,
                        bool acceleration_constraints,
                        int K,
                        int n,
                        float delta_t,
                        py::array_t<double> p_min_npy,
                        py::array_t<double> p_max_npy,
                        float w_g_p,
                        float w_g_v,
                        float w_s,
                        float v_bar,
                        float f_bar) {

                            // convert waypoints from numpy to eigen
                            py::array_t<double> array = waypoints_npy.cast<py::array_t<double>>();
                            auto buffer = array.request();
                            Eigen::MatrixXd waypoints = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<double *>(buffer.ptr), buffer.shape[0], buffer.shape[1]);

                            // convert initial position from numpy to eigen
                            array = initial_pos_npy.cast<py::array_t<double>>();
                            buffer = array.request();
                            Eigen::VectorXd initial_pos = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[0]);

                            // convert p_min, p_max from numpy to eigen
                            array = p_min_npy.cast<py::array_t<double>>();
                            buffer = array.request();
                            Eigen::VectorXd p_min = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[1]);
                            array = p_max_npy.cast<py::array_t<double>>();
                            buffer = array.request();
                            Eigen::VectorXd p_max = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[1]);
                            
                            return new Drone(params_filepath, waypoints, initial_pos, hard_waypoint_constraints,
                                            acceleration_constraints, K, n, delta_t, p_min, p_max, w_g_p, w_g_v,
                                            w_s, v_bar, f_bar);}),
            py::arg("params_filepath"), py::arg("waypoints"), py::arg("initial_pos"), py::arg("hard_waypoint_constraints"),
            py::arg("acceleration_constraints"), py::arg("K"), py::arg("n"), py::arg("delta_t"), py::arg("p_min"),
            py::arg("p_max"), py::arg("w_g_p"), py::arg("w_g_v"), py::arg("w_s"), py::arg("v_bar"), py::arg("f_bar"))
        .def("solve", [](Drone &instance, double current_time, py::array_t<double> x_0_npy, int j, py::list thetas_py, py::array_t<double> xi_npy)
            {
                // convert current state from numpy to eigen
                py::array_t<double> array = x_0_npy.cast<py::array_t<double>>();
                auto buffer = array.request();
                Eigen::VectorXd x_0 = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[0]);

                // convert thetas to std::vector of Eigen::SparseMatrix<double>
                std::vector<Eigen::SparseMatrix<double>> thetas;
                for (int i = 0; i < thetas_py.size(); ++i) {
                    py::array_t<double> array = thetas_py[i].cast<py::array_t<double>>();
                    auto buffer = array.request();
                    Eigen::MatrixXd theta_dense = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<double *>(buffer.ptr), buffer.shape[0], buffer.shape[1]);
                    Eigen::SparseMatrix<double> theta = theta_dense.sparseView();
                    thetas.push_back(theta);
                }

                // convert xi from numpy to Eigen::VectorXd
                array = xi_npy.cast<py::array_t<double>>();
                buffer = array.request();
                Eigen::VectorXd xi = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[0]);
                return instance.solve(current_time, x_0, j, thetas, xi);
            });
    
    py::class_<Swarm>(m, "Swarm")
        .def(py::init<std::vector<Drone>>())
        .def(py::init([](py::list drone_list) {
            std::vector<Drone> drones;
            for (auto item : drone_list) {
                drones.push_back(item.cast<Drone>());
            }

            return new Swarm(drones);
        }))
        .def("solve", [](Swarm &instance, double current_time, py::list x_0_vector_py, py::list prev_trajectories_py) {
            std::vector<Eigen::VectorXd> prev_trajectories;
            for (auto item : prev_trajectories_py) {
                py::array_t<double> array = item.cast<py::array_t<double>>();
                auto buffer = array.request();
                Eigen::VectorXd prev_trajectory(buffer.shape[0]);

                for (ssize_t i = 0; i < array.size(); ++i) {
                    prev_trajectory(i) = array.at(i);
                }

                // Eigen::VectorXd prev_trajectory = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[0]);
                prev_trajectories.push_back(prev_trajectory);
            }
            std::vector<Eigen::VectorXd> x_0_vector;
            for (auto item : x_0_vector_py) {
                py::array_t<double> array = item.cast<py::array_t<double>>();
                auto buffer = array.request();
                Eigen::VectorXd x_0(buffer.shape[0]);

                // Iterate over the array and copy each element
                // I tried doing with buffer and there were extremely difficult memory bugs
                for (ssize_t i = 0; i < array.size(); ++i) {
                    x_0(i) = array.at(i);
                }
                x_0_vector.push_back(x_0);
            }
            return instance.solve(current_time, x_0_vector, prev_trajectories);
        })
        .def("run_simulation", [](Swarm &instance) {
            Swarm::SwarmResult swarm_result = instance.runSimulation();
            return swarm_result;
        });
}