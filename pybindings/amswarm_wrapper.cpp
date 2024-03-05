#include <iostream>
#include <swarm.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(amswarm, m)
{
    py::class_<DroneResult>(m, "DroneResult")
        .def(py::init<>())
        .def_readwrite("control_input_trajectory_vector", &DroneResult::control_input_trajectory_vector)
        .def_readwrite("state_trajectory_vector", &DroneResult::state_trajectory_vector)
        .def_readwrite("position_trajectory_vector", &DroneResult::position_trajectory_vector)
        .def_readwrite("control_input_trajectory", &DroneResult::control_input_trajectory)
        .def_readwrite("state_trajectory", &DroneResult::state_trajectory)
        .def_readwrite("position_trajectory", &DroneResult::position_trajectory)
        .def_readwrite("spline_coeffs", &DroneResult::spline_coeffs);

    py::class_<DroneSolveArgs>(m, "DroneSolveArgs")
        .def(py::init<>())
        .def_readwrite("current_time", &DroneSolveArgs::current_time)
        .def_readwrite("num_obstacles", &DroneSolveArgs::num_obstacles)
        .def_readwrite("obstacle_envelopes", &DroneSolveArgs::obstacle_envelopes)
        .def_readwrite("obstacle_positions", &DroneSolveArgs::obstacle_positions)
        .def_readwrite("x_0", &DroneSolveArgs::x_0);

    py::class_<Drone::MPCWeights>(m, "MPCWeights")
        .def(py::init<double, double, double, double, double, double, double>(), 
            py::arg("waypoint_pos") = 7000, 
            py::arg("waypoint_vel") = 1000,
            py::arg("waypoint_acc") = 100, 
            py::arg("smoothness") = 100, 
            py::arg("input_smoothness") = 1000, 
            py::arg("input_continuity") = 100)
        .def(py::init<>())
        .def_readwrite("waypoint_pos", &Drone::MPCWeights::waypoint_pos)
        .def_readwrite("waypoint_vel", &Drone::MPCWeights::waypoint_vel)
        .def_readwrite("waypoint_acc", &Drone::MPCWeights::waypoint_acc)
        .def_readwrite("smoothness", &Drone::MPCWeights::smoothness)
        .def_readwrite("input_smoothness", &Drone::MPCWeights::input_smoothness)
        .def_readwrite("input_continuity", &Drone::MPCWeights::input_continuity)
        .def(py::pickle(
            [](const Drone::MPCWeights &w) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(w.waypoint_pos, w.waypoint_vel, w.waypoint_acc, w.smoothness, w.input_smoothness, w.input_continuity);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::MPCWeights w;
                w.waypoint_pos = t[0].cast<double>();
                w.waypoint_vel = t[1].cast<double>();
                w.waypoint_acc = t[2].cast<double>();
                w.smoothness = t[3].cast<double>();
                w.input_smoothness = t[4].cast<double>();
                w.input_continuity = t[5].cast<double>();
                return w;
            }));

    py::class_<Drone::MPCConfig>(m, "MPCConfig")
        .def(py::init<int, int, double, double>(), 
            py::arg("K") = 25, 
            py::arg("n") = 10, 
            py::arg("delta_t") = 1.0 / 8.0,
            py::arg("bf_gamma") = 1.0)
        .def(py::init<>())
        .def_readwrite("K", &Drone::MPCConfig::K)
        .def_readwrite("n", &Drone::MPCConfig::n)
        .def_readwrite("delta_t", &Drone::MPCConfig::delta_t)
        .def_readwrite("bf_gamma", &Drone::MPCConfig::bf_gamma)
        .def(py::pickle(
            [](const Drone::MPCConfig &c) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(c.K, c.n, c.delta_t, c.bf_gamma);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::MPCConfig c;
                c.K = t[0].cast<int>();
                c.n = t[1].cast<int>();
                c.delta_t = t[2].cast<double>();
                c.bf_gamma = t[3].cast<double>();
                return c;
            }));

    py::class_<Drone::PhysicalLimits>(m, "PhysicalLimits")
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd, double, double>(), 
            py::arg("p_min") = Eigen::VectorXd::Constant(3, -10), 
            py::arg("p_max") = Eigen::VectorXd::Constant(3, 10), 
            py::arg("v_bar") = 1.73, 
            py::arg("a_bar") = 0.75 * 9.81)
        .def(py::init<>())
        .def_readwrite("p_min", &Drone::PhysicalLimits::p_min)
        .def_readwrite("p_max", &Drone::PhysicalLimits::p_max)
        .def_readwrite("v_bar", &Drone::PhysicalLimits::v_bar)
        .def_readwrite("a_bar", &Drone::PhysicalLimits::a_bar)
        .def(py::pickle(
            [](const Drone::PhysicalLimits &l) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(l.p_min, l.p_max, l.v_bar, l.a_bar);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::PhysicalLimits l;
                l.p_min = t[0].cast<Eigen::VectorXd>();
                l.p_max = t[1].cast<Eigen::VectorXd>();
                l.v_bar = t[2].cast<double>();
                l.a_bar = t[3].cast<double>();
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

    py::class_<Drone>(m, "Drone")
        .def(py::init<Eigen::MatrixXd, Drone::MPCConfig, Drone::MPCWeights, Drone::PhysicalLimits, Drone::SparseDynamics, Eigen::VectorXd>(),
            py::arg("waypoints"), py::arg("config"), py::arg("weights"), py::arg("limits"), py::arg("dynamics"), py::arg("initial_pos"))
        .def("solve", &Drone::solve, py::arg("args"));
    
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
                        py::list prev_trajectories_py, py::list prev_inputs_py) {
            
            std::vector<Eigen::VectorXd> prev_inputs;
            if (!prev_inputs_py.empty()) {
                for (auto item : prev_inputs_py) {
                    py::array_t<double> array = item.cast<py::array_t<double>>();
                    auto buffer = array.request();
                    Eigen::VectorXd prev_input(buffer.shape[0]);

                    for (ssize_t i = 0; i < array.size(); ++i) {
                        prev_input(i) = array.at(i);
                    }
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

            // Call the solve method with or without prev_inputs
            if (prev_inputs.empty()) {
                return instance.solve(current_time, x_0_vector, prev_trajectories, opt);
            } else {
                return instance.solve(current_time, x_0_vector, prev_trajectories, opt, prev_inputs);
            }
        }, py::arg("current_time"), py::arg("x_0_vector"), py::arg("prev_trajectories"), py::arg("prev_inputs") = py::list());
}