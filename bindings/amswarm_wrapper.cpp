#include <iostream>
#include <amswarm/swarm.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(amswarm, m)
{
    py::class_<DroneResult>(m, "DroneResult")
        .def(py::init<>())
        .def_readwrite("state_trajectory_vector", &DroneResult::state_trajectory_vector)
        .def_readwrite("position_trajectory_vector", &DroneResult::position_trajectory_vector)
        .def_readwrite("state_trajectory", &DroneResult::state_trajectory)
        .def_readwrite("position_trajectory", &DroneResult::position_trajectory)
        .def_readwrite("input_position_trajectory_vector", &DroneResult::input_position_trajectory_vector)
        .def_readwrite("input_velocity_trajectory_vector", &DroneResult::input_velocity_trajectory_vector)
        .def_readwrite("input_acceleration_trajectory_vector", &DroneResult::input_acceleration_trajectory_vector)
        .def_readwrite("input_position_trajectory", &DroneResult::input_position_trajectory)
        .def_readwrite("input_velocity_trajectory", &DroneResult::input_velocity_trajectory)
        .def_readwrite("input_acceleration_trajectory", &DroneResult::input_acceleration_trajectory)
        .def_readwrite("spline_coeffs", &DroneResult::spline_coeffs)
        .def_static("generateInitialDroneResult", &DroneResult::generateInitialDroneResult, 
                    py::arg("initial_position"), py::arg("K"))
        .def("advanceForNextSolveStep", &DroneResult::advanceForNextSolveStep);

    py::class_<ConstraintConfig>(m, "ConstraintConfig")
        .def(py::init<>())
        .def_readwrite("enable_waypoints_pos_constraint", &ConstraintConfig::enable_waypoints_pos_constraint)
        .def_readwrite("enable_waypoints_vel_constraint", &ConstraintConfig::enable_waypoints_vel_constraint)
        .def_readwrite("enable_waypoints_acc_constraint", &ConstraintConfig::enable_waypoints_acc_constraint)
        .def_readwrite("enable_input_continuity_constraint", &ConstraintConfig::enable_input_continuity_constraint)
        .def("setWaypointsConstraints", &ConstraintConfig::setWaypointsConstraints, 
            py::arg("pos"), py::arg("vel"), py::arg("acc"))
        .def("setInputContinuityConstraints", &ConstraintConfig::setInputContinuityConstraints, 
            py::arg("flag"));

    py::class_<DroneSolveArgs>(m, "DroneSolveArgs")
        .def(py::init<>())
        .def_readwrite("current_time", &DroneSolveArgs::current_time)
        .def_readwrite("num_obstacles", &DroneSolveArgs::num_obstacles)
        .def_readwrite("obstacle_envelopes", &DroneSolveArgs::obstacle_envelopes)
        .def_readwrite("obstacle_positions", &DroneSolveArgs::obstacle_positions)
        .def_readwrite("x_0", &DroneSolveArgs::x_0)
        .def_readwrite("u_0", &DroneSolveArgs::u_0)
        .def_readwrite("u_dot_0", &DroneSolveArgs::u_dot_0)
        .def_readwrite("u_ddot_0", &DroneSolveArgs::u_ddot_0)
        .def_readwrite("constraintConfig", &DroneSolveArgs::constraintConfig);

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
        .def(py::init<int, int, double, double, double, double, double, double, double, double, double, double>(), 
            py::arg("K") = 25, 
            py::arg("n") = 10, 
            py::arg("mpc_freq") = 8.0,
            py::arg("bf_gamma") = 1.0,
            py::arg("waypoints_pos_tol") = 1e-2,
            py::arg("waypoints_vel_tol") = 1e-2,
            py::arg("waypoints_acc_tol") = 1e-2,
            py::arg("input_continuity_tol") = 1e-2,
            py::arg("pos_tol") = 1e-2,
            py::arg("vel_tol") = 1e-2,
            py::arg("acc_tol") = 1e-2,
            py::arg("collision_tol") = 1e-2)
        .def(py::init<>())
        .def_readwrite("K", &Drone::MPCConfig::K)
        .def_readwrite("n", &Drone::MPCConfig::n)
        .def_readwrite("mpc_freq", &Drone::MPCConfig::mpc_freq)
        .def_readwrite("bf_gamma", &Drone::MPCConfig::bf_gamma)
        .def_readwrite("waypoints_pos_tol", &Drone::MPCConfig::waypoints_pos_tol)
        .def_readwrite("waypoints_vel_tol", &Drone::MPCConfig::waypoints_vel_tol)
        .def_readwrite("waypoints_acc_tol", &Drone::MPCConfig::waypoints_acc_tol)
        .def_readwrite("input_continuity_tol", &Drone::MPCConfig::input_continuity_tol)
        .def_readwrite("pos_tol", &Drone::MPCConfig::pos_tol)
        .def_readwrite("vel_tol", &Drone::MPCConfig::vel_tol)
        .def_readwrite("acc_tol", &Drone::MPCConfig::acc_tol)
        .def_readwrite("collision_tol", &Drone::MPCConfig::collision_tol)
        .def(py::pickle(
            [](const Drone::MPCConfig &c) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(c.K, c.n, c.mpc_freq, c.bf_gamma, c.waypoints_pos_tol, c.waypoints_vel_tol, c.waypoints_acc_tol, c.input_continuity_tol, c.pos_tol, c.vel_tol, c.acc_tol, c.collision_tol);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 12)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::MPCConfig c;
                c.K = t[0].cast<int>();
                c.n = t[1].cast<int>();
                c.mpc_freq = t[2].cast<double>();
                c.bf_gamma = t[3].cast<double>();
                c.waypoints_pos_tol = t[4].cast<double>();
                c.waypoints_vel_tol = t[5].cast<double>();
                c.waypoints_acc_tol = t[6].cast<double>();
                c.input_continuity_tol = t[7].cast<double>();
                c.pos_tol = t[8].cast<double>();
                c.vel_tol = t[9].cast<double>();
                c.acc_tol = t[10].cast<double>();
                c.collision_tol = t[11].cast<double>();
                return c;
            }));

    py::class_<Drone::PhysicalLimits>(m, "PhysicalLimits")
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd, double, double, double, double, double>(), 
            py::arg("p_min") = Eigen::VectorXd::Constant(3, -10), 
            py::arg("p_max") = Eigen::VectorXd::Constant(3, 10), 
            py::arg("v_bar") = 1.73, 
            py::arg("a_bar") = 0.75 * 9.81,
            py::arg("x_collision_envelope") = 0.25,
            py::arg("y_collision_envelope") = 0.25,
            py::arg("z_collision_envelope") = 2.0 / 3.0)
        .def(py::init<>())
        .def_readwrite("p_min", &Drone::PhysicalLimits::p_min)
        .def_readwrite("p_max", &Drone::PhysicalLimits::p_max)
        .def_readwrite("v_bar", &Drone::PhysicalLimits::v_bar)
        .def_readwrite("a_bar", &Drone::PhysicalLimits::a_bar)
        .def_readwrite("x_collision_envelope", &Drone::PhysicalLimits::x_collision_envelope)
        .def_readwrite("y_collision_envelope", &Drone::PhysicalLimits::y_collision_envelope)
        .def_readwrite("z_collision_envelope", &Drone::PhysicalLimits::z_collision_envelope)
        .def(py::pickle(
            [](const Drone::PhysicalLimits &l) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(l.p_min, l.p_max, l.v_bar, l.a_bar, l.x_collision_envelope, l.y_collision_envelope, l.z_collision_envelope);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 7)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Drone::PhysicalLimits l;
                l.p_min = t[0].cast<Eigen::VectorXd>();
                l.p_max = t[1].cast<Eigen::VectorXd>();
                l.v_bar = t[2].cast<double>();
                l.a_bar = t[3].cast<double>();
                l.x_collision_envelope = t[4].cast<double>();
                l.y_collision_envelope = t[5].cast<double>();
                l.z_collision_envelope = t[6].cast<double>();
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

    py::class_<AMSolverConfig>(m, "AMSolverConfig")
        .def(py::init<double, double, int>(),
            py::arg("rho_init") = 1.3, 
            py::arg("max_rho") = 5.0e5, 
            py::arg("max_iters") = 1000)
        .def(py::init<>())
        .def_readwrite("rho_init", &AMSolverConfig::rho_init)
        .def_readwrite("max_rho", &AMSolverConfig::max_rho)
        .def_readwrite("max_iters", &AMSolverConfig::max_iters)
        .def(py::pickle(
            [](const AMSolverConfig &c) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(c.rho_init, c.max_rho, c.max_iters);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                AMSolverConfig c;
                c.rho_init = t[0].cast<double>();
                c.max_rho = t[1].cast<double>();
                c.max_iters = t[2].cast<int>();
                return c;
            }));

    py::class_<Drone, std::shared_ptr<Drone>>(m, "Drone")
        .def(py::init<AMSolverConfig, Eigen::MatrixXd, Drone::MPCConfig, Drone::MPCWeights, Drone::PhysicalLimits, Drone::SparseDynamics>(),
            py::arg("solverConfig"), py::arg("waypoints"), py::arg("mpcConfig"), py::arg("weights"), py::arg("limits"), py::arg("dynamics"))
        .def("solve", &Drone::solve, py::arg("args"));
    
    py::class_<Swarm>(m, "Swarm")
        .def(py::init([](const std::vector<std::shared_ptr<Drone>>& drones) {
            // Directly use the shared_ptr vector without manual casting
            return std::make_unique<Swarm>(drones);
        }))
        .def("solve", &Swarm::solve, 
            py::arg("current_time"), 
            py::arg("initial_states"),
            py::arg("previous_results"),
            py::arg("constraint_configs"));
}