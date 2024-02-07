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
        .def_readwrite("spline_coeffs", &Drone::DroneResult::spline_coeffs)
        .def_readwrite("is_successful", &Drone::DroneResult::is_successful);

    py::class_<Drone::SolveOptions>(m, "SolveOptions") // TODO set defaults?
        .def(py::init<>())
        .def_readwrite("waypoint_position_constraints", &Drone::SolveOptions::waypoint_position_constraints)
        .def_readwrite("waypoint_velocity_constraints", &Drone::SolveOptions::waypoint_velocity_constraints)
        .def_readwrite("waypoint_acceleration_constraints", &Drone::SolveOptions::waypoint_acceleration_constraints)
        .def_readwrite("input_continuity_constraints", &Drone::SolveOptions::input_continuity_constraints)
        .def_readwrite("input_dot_continuity_constraints", &Drone::SolveOptions::input_dot_continuity_constraints)
        .def_readwrite("input_ddot_continuity_constraints", &Drone::SolveOptions::input_ddot_continuity_constraints);

    py::class_<Drone::MPCWeights>(m, "MPCWeights")
        .def(py::init<double, double, double, double, double, double, double>(), 
            py::arg("w_goal_pos") = 7000, 
            py::arg("w_goal_vel") = 1000, 
            py::arg("w_smoothness") = 100, 
            py::arg("w_input_smoothness") = 1000, 
            py::arg("w_input_continuity") = 100, 
            py::arg("w_input_dot_continuity") = 100, 
            py::arg("w_input_ddot_continuity") = 100)
        .def(py::init<>())
        .def_readwrite("w_goal_pos", &Drone::MPCWeights::w_goal_pos)
        .def_readwrite("w_goal_vel", &Drone::MPCWeights::w_goal_vel)
        .def_readwrite("w_smoothness", &Drone::MPCWeights::w_smoothness)
        .def_readwrite("w_input_smoothness", &Drone::MPCWeights::w_input_smoothness)
        .def_readwrite("w_input_continuity", &Drone::MPCWeights::w_input_continuity)
        .def_readwrite("w_input_dot_continuity", &Drone::MPCWeights::w_input_dot_continuity)
        .def_readwrite("w_input_ddot_continuity", &Drone::MPCWeights::w_input_ddot_continuity)
        .def(py::pickle(
            [](const Drone::MPCWeights &w) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(w.w_goal_pos, w.w_goal_vel, w.w_smoothness, w.w_input_smoothness, w.w_input_continuity, w.w_input_dot_continuity, w.w_input_ddot_continuity);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 7)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::MPCWeights w;
                w.w_goal_pos = t[0].cast<double>();
                w.w_goal_vel = t[1].cast<double>();
                w.w_smoothness = t[2].cast<double>();
                w.w_input_smoothness = t[3].cast<double>();
                w.w_input_continuity = t[4].cast<double>();
                w.w_input_dot_continuity = t[5].cast<double>();
                w.w_input_ddot_continuity = t[6].cast<double>();
                return w;
            }));

    py::class_<Drone::MPCConfig>(m, "MPCConfig")
        .def(py::init<int, int, double>(), 
            py::arg("K") = 25, 
            py::arg("n") = 10, 
            py::arg("delta_t") = 1.0 / 8.0)
        .def(py::init<>())
        .def_readwrite("K", &Drone::MPCConfig::K)
        .def_readwrite("n", &Drone::MPCConfig::n)
        .def_readwrite("delta_t", &Drone::MPCConfig::delta_t)
        .def(py::pickle(
            [](const Drone::MPCConfig &c) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(c.K, c.n, c.delta_t);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::MPCConfig c;
                c.K = t[0].cast<int>();
                c.n = t[1].cast<int>();
                c.delta_t = t[2].cast<double>();
                return c;
            }));

    py::class_<Drone::PhysicalLimits>(m, "PhysicalLimits")
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd, double, double>(), 
            py::arg("p_min") = Eigen::VectorXd::Constant(3, -10), 
            py::arg("p_max") = Eigen::VectorXd::Constant(3, 10), 
            py::arg("v_bar") = 1.73, 
            py::arg("f_bar") = 0.75 * 9.81)
        .def(py::init<>())
        .def_readwrite("p_min", &Drone::PhysicalLimits::p_min)
        .def_readwrite("p_max", &Drone::PhysicalLimits::p_max)
        .def_readwrite("v_bar", &Drone::PhysicalLimits::v_bar)
        .def_readwrite("f_bar", &Drone::PhysicalLimits::f_bar)
        .def(py::pickle(
            [](const Drone::PhysicalLimits &l) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(l.p_min, l.p_max, l.v_bar, l.f_bar);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::PhysicalLimits l;
                l.p_min = t[0].cast<Eigen::VectorXd>();
                l.p_max = t[1].cast<Eigen::VectorXd>();
                l.v_bar = t[2].cast<double>();
                l.f_bar = t[3].cast<double>();
                return l;
            }));

    py::class_<Drone::SparseDynamics>(m, "SparseDynamics")
        .def(py::init<const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&, 
                   const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&>(),
                   py::arg("A"), py::arg("B"), py::arg("A_prime"), py::arg("B_prime"))
        .def(py::init<>())
        .def_readwrite("A", &Drone::SparseDynamics::A)
        .def_readwrite("B", &Drone::SparseDynamics::B)
        .def_readwrite("A_prime", &Drone::SparseDynamics::A_prime)
        .def_readwrite("B_prime", &Drone::SparseDynamics::B_prime)
        .def(py::pickle(
            [](const Drone::SparseDynamics &d) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(d.A, d.B, d.A_prime, d.B_prime);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::SparseDynamics d;
                d.A = t[0].cast<Eigen::SparseMatrix<double>>();
                d.B = t[1].cast<Eigen::SparseMatrix<double>>();
                d.A_prime = t[2].cast<Eigen::SparseMatrix<double>>();
                d.B_prime = t[3].cast<Eigen::SparseMatrix<double>>();
                return d;
            }));

    // Binding for SwarmResult
    py::class_<Swarm::SwarmResult>(m, "SwarmResult")
        .def(py::init<>())
        .def_readwrite("drone_results", &Swarm::SwarmResult::drone_results)
        .def("getDroneData", &Swarm::SwarmResult::getDroneResult);  

    py::class_<Drone>(m, "Drone")
        .def(py::init<Eigen::MatrixXd, Drone::MPCConfig, Drone::MPCWeights, Drone::PhysicalLimits, Drone::SparseDynamics, Eigen::VectorXd>())
        .def(py::init([](py::array_t<double> waypoints_npy,
                        Drone::MPCConfig config,
                        Drone::MPCWeights weights,
                        Drone::PhysicalLimits limits,
                        Drone::SparseDynamics dynamics,
                        py::array_t<double> initial_pos_npy) {

                            // convert waypoints from numpy to eigen
                            py::array_t<double> array = waypoints_npy.cast<py::array_t<double>>();
                            auto buffer = array.request();
                            Eigen::MatrixXd waypoints = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<double *>(buffer.ptr), buffer.shape[0], buffer.shape[1]);

                            // convert initial position from numpy to eigen
                            array = initial_pos_npy.cast<py::array_t<double>>();
                            buffer = array.request();
                            Eigen::VectorXd initial_pos = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[0]);
                            
                            return new Drone(waypoints, config, weights, limits, dynamics, initial_pos);}),
            py::arg("waypoints"), py::arg("config"), py::arg("weights"), py::arg("limits"), py::arg("dynamics"), py::arg("initial_pos"))
        .def("solve", [](Drone &instance, double current_time,
                        py::array_t<double> x_0_npy, py::array_t<double> initial_guess_control_input_trajectory_vector_py,
                        int j, py::list thetas_py, py::array_t<double> xi_npy,
                        Drone::SolveOptions opt)
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
                return instance.solve(current_time, x_0,
                                    j, thetas, xi, opt, initial_guess_control_input_trajectory_vector);
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
                        py::list prev_trajectories_py,
                        py::list opt_py, py::list prev_inputs_py) {
            
            std::vector<Eigen::VectorXd> prev_inputs;
            if (!prev_inputs_py.empty()) {
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

            std::vector<Drone::SolveOptions> opt;
            for (auto item : opt_py) {
                opt.push_back(item.cast<Drone::SolveOptions>());
            }

            // Call the solve method with or without prev_inputs
            if (prev_inputs.empty()) {
                return instance.solve(current_time, x_0_vector, prev_trajectories, opt);
            } else {
                return instance.solve(current_time, x_0_vector, prev_trajectories, opt, prev_inputs);
            }
        }, py::arg("current_time"), py::arg("x_0_vector"), py::arg("prev_trajectories"), py::arg("opt"), py::arg("prev_inputs") = py::list())
        .def("run_simulation", [](Swarm &instance) {
            Swarm::SwarmResult swarm_result = instance.runSimulation();
            return swarm_result;
        });
}