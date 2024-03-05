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
        .def_readwrite("x_0", &DroneSolveArgs::x_0)
        .def_readwrite("u_0", &DroneSolveArgs::u_0)
        .def_readwrite("u_dot_0", &DroneSolveArgs::u_dot_0)
        .def_readwrite("u_ddot_0", &DroneSolveArgs::u_ddot_0);

    py::class_<Drone::MPCWeights>(m, "MPCWeights")
        .def(py::init<double, double, double, double, double, double>(), 
            py::arg("waypoints_pos") = 7000, 
            py::arg("waypoints_vel") = 1000,
            py::arg("waypoints_acc") = 100, 
            py::arg("smoothness") = 100, 
            py::arg("input_smoothness") = 1000, 
            py::arg("input_continuity") = 100)
        .def(py::init<>())
        .def_readwrite("waypoints_pos", &Drone::MPCWeights::waypoints_pos)
        .def_readwrite("waypoints_vel", &Drone::MPCWeights::waypoints_vel)
        .def_readwrite("waypoints_acc", &Drone::MPCWeights::waypoints_acc)
        .def_readwrite("smoothness", &Drone::MPCWeights::smoothness)
        .def_readwrite("input_smoothness", &Drone::MPCWeights::input_smoothness)
        .def_readwrite("input_continuity", &Drone::MPCWeights::input_continuity)
        .def(py::pickle(
            [](const Drone::MPCWeights &w) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(w.waypoints_pos, w.waypoints_vel, w.waypoints_acc, w.smoothness, w.input_smoothness, w.input_continuity);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::MPCWeights w;
                w.waypoints_pos = t[0].cast<double>();
                w.waypoints_vel = t[1].cast<double>();
                w.waypoints_acc = t[2].cast<double>();
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

    py::class_<Drone, std::shared_ptr<Drone>>(m, "Drone")
        .def(py::init<Eigen::MatrixXd, Drone::MPCConfig, Drone::MPCWeights, Drone::PhysicalLimits, Drone::SparseDynamics, Eigen::VectorXd>(),
            py::arg("waypoints"), py::arg("config"), py::arg("weights"), py::arg("limits"), py::arg("dynamics"), py::arg("initial_pos"))
        .def("solve", &Drone::solve, py::arg("args"));
    
    py::class_<Swarm>(m, "Swarm")
        .def(py::init([](py::list drones_py) {
            std::vector<std::unique_ptr<Drone>> drones;
            for (py::handle drone_py : drones_py) {
                Drone* drone = drone_py.cast<Drone*>();
                drones.push_back(std::unique_ptr<Drone>(drone));
            }
            return new Swarm(std::move(drones));
        }))
        .def("solve", &Swarm::solve, 
            py::arg("current_time"), 
            py::arg("x_0_vector"), 
            py::arg("prev_trajectories"), 
            py::arg("prev_inputs") = py::list());
}