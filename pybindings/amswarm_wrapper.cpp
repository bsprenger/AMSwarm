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
        .def_readwrite("position_trajectory", &Drone::DroneResult::position_trajectory)
        .def_readwrite("is_successful", &Drone::DroneResult::is_successful);

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
        .def(py::init<std::string&, Eigen::MatrixXd, Eigen::VectorXd, int, int, float, Eigen::VectorXd, Eigen::VectorXd, float, float, float, float, float, float, float, float, float>())
        .def(py::init([](std::string& params_filepath,
                        py::array_t<double> waypoints_npy,
                        py::array_t<double> initial_pos_npy,
                        int K,
                        int n,
                        float delta_t,
                        py::array_t<double> p_min_npy,
                        py::array_t<double> p_max_npy,
                        float w_goal_pos,
                        float w_goal_vel,
                        float w_smoothness,
                        float w_input_smoothness,
                        float w_input_continuity,
                        float w_input_dot_continuity,
                        float w_input_ddot_continuity,
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
                            
                            return new Drone(params_filepath, waypoints, initial_pos, K, n, delta_t, p_min, p_max, w_goal_pos, w_goal_vel,
                                            w_smoothness, w_input_smoothness, w_input_continuity, w_input_dot_continuity, w_input_ddot_continuity, v_bar, f_bar);}),
            py::arg("params_filepath"), py::arg("waypoints"), py::arg("initial_pos"), py::arg("K"), py::arg("n"), py::arg("delta_t"), py::arg("p_min"),
            py::arg("p_max"), py::arg("w_goal_pos"), py::arg("w_goal_vel"), py::arg("w_smoothness"), py::arg("w_input_smoothness"), py::arg("w_input_continuity"), py::arg("w_input_dot_continuity"), py::arg("w_input_ddot_continuity"), py::arg("v_bar"), py::arg("f_bar"))
        .def("solve", [](Drone &instance, double current_time,
                        py::array_t<double> x_0_npy, py::array_t<double> initial_guess_control_input_trajectory_vector_py,
                        int j, py::list thetas_py, py::array_t<double> xi_npy,
                        bool waypoint_position_constraints,
                        bool waypoint_velocity_constraints,
                        bool waypoint_acceleration_constraints,
                        bool input_continuity_constraints,
                        bool input_dot_continuity_constraints,
                        bool input_ddot_continuity_constraints)
            {
                // convert current state from numpy to eigen
                py::array_t<double> array = x_0_npy.cast<py::array_t<double>>();
                auto buffer = array.request();
                Eigen::VectorXd x_0 = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[0]);

                // convert initial guess from numpy to eigen
                array = initial_guess_control_input_trajectory_vector_py.cast<py::array_t<double>>();
                buffer = array.request();
                Eigen::VectorXd initial_guess_control_input_trajectory_vector = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[0]);

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
                return instance.solve(current_time, x_0, initial_guess_control_input_trajectory_vector,
                                    j, thetas, xi, waypoint_position_constraints,
                                    waypoint_velocity_constraints,
                                    waypoint_acceleration_constraints,
                                    input_continuity_constraints,
                                    input_dot_continuity_constraints,
                                    input_ddot_continuity_constraints);
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
        .def("solve", [](Swarm &instance, double current_time, py::list x_0_vector_py,
                        py::list prev_inputs_py, py::list prev_trajectories_py,
                        py::list waypoint_position_constraints_py,
                        py::list waypoint_velocity_constraints_py,
                        py::list waypoint_acceleration_constraints_py,
                        py::list input_continuity_constraints_py,
                        py::list input_dot_continuity_constraints_py,
                        py::list input_ddot_continuity_constraints_py) {
            std::vector<Eigen::VectorXd> prev_inputs;
            for (auto item : prev_inputs_py) {
                py::array_t<double> array = item.cast<py::array_t<double>>();
                auto buffer = array.request();
                Eigen::VectorXd prev_input(buffer.shape[0]);

                for (ssize_t i = 0; i < array.size(); ++i) {
                    prev_input(i) = array.at(i);
                }

                // Eigen::VectorXd prev_trajectory = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[0]);
                prev_inputs.push_back(prev_input);
            }
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
            std::vector<bool> waypoint_position_constraints;
            for (auto item : waypoint_position_constraints_py) {
                waypoint_position_constraints.push_back(item.cast<bool>());
            }
            std::vector<bool> waypoint_velocity_constraints;
            for (auto item : waypoint_velocity_constraints_py) {
                waypoint_velocity_constraints.push_back(item.cast<bool>());
            }
            std::vector<bool> waypoint_acceleration_constraints;
            for (auto item : waypoint_acceleration_constraints_py) {
                waypoint_acceleration_constraints.push_back(item.cast<bool>());
            }
            std::vector<bool> input_continuity_constraints;
            for (auto item : input_continuity_constraints_py) {
                input_continuity_constraints.push_back(item.cast<bool>());
            }
            std::vector<bool> input_dot_continuity_constraints;
            for (auto item : input_dot_continuity_constraints_py) {
                input_dot_continuity_constraints.push_back(item.cast<bool>());
            }
            std::vector<bool> input_ddot_continuity_constraints;
            for (auto item : input_ddot_continuity_constraints_py) {
                input_ddot_continuity_constraints.push_back(item.cast<bool>());
            }
            return instance.solve(current_time, x_0_vector, prev_inputs, prev_trajectories,
                                waypoint_position_constraints,
                                waypoint_velocity_constraints,
                                waypoint_acceleration_constraints,
                                input_continuity_constraints,
                                input_dot_continuity_constraints,
                                input_ddot_continuity_constraints);
        })
        .def("run_simulation", [](Swarm &instance) {
            Swarm::SwarmResult swarm_result = instance.runSimulation();
            return swarm_result;
        });
}