#include <iostream>
#include <simulator.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::map<int, Eigen::MatrixXd> dictToMatrixMap(py::dict d)
{
    std::map<int, Eigen::MatrixXd> result;

    for (const auto &item : d)
    {
        int key = item.first.cast<int>();

        // Extract the NumPy array from the Python object
        py::array_t<double> array = item.second.cast<py::array_t<double>>();
        auto buffer = array.request();

        // Map the NumPy array to Eigen::MatrixXd
        Eigen::MatrixXd value = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<double *>(buffer.ptr), buffer.shape[0], buffer.shape[1]);

        result[key] = value;
    }

    return result;
}

std::map<int, Eigen::VectorXd> dictToVectorMap(py::dict d)
{
    std::map<int, Eigen::VectorXd> result;

    for (const auto &item : d)
    {
        int key = item.first.cast<int>();

        // Extract the NumPy array from the Python object
        py::array_t<double> array = item.second.cast<py::array_t<double>>();
        auto buffer = array.request();

        // Map the NumPy array to Eigen::MatrixXd
        Eigen::VectorXd value = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(buffer.ptr), buffer.shape[0]);

        result[key] = value;
    }

    return result;
}

PYBIND11_MODULE(amswarm, m)
{
    py::class_<Drone::OptimizationResult>(m, "OptimizationResult")
        .def(py::init<>())
        .def_readwrite("input_traj_vector", &Drone::OptimizationResult::input_traj_vector)
        .def_readwrite("state_traj_vector", &Drone::OptimizationResult::state_traj_vector)
        .def_readwrite("pos_traj_vector", &Drone::OptimizationResult::pos_traj_vector)
        .def_readwrite("input_traj_matrix", &Drone::OptimizationResult::input_traj_matrix)
        .def_readwrite("state_traj_matrix", &Drone::OptimizationResult::state_traj_matrix)
        .def_readwrite("pos_traj_matrix", &Drone::OptimizationResult::pos_traj_matrix);     

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<int, int, int, float, Eigen::VectorXd, Eigen::VectorXd, float, float, float, int, float, float, std::map<int, Eigen::VectorXd>, std::map<int, Eigen::MatrixXd>, std::string &>())
        .def(py::init([](int num_drones,
                         int K, int n, float delta_t, py::array_t<double> p_min_npy, py::array_t<double> p_max_npy, float w_g_p, float w_g_v, float w_s, int kappa, float v_bar, float f_bar, py::dict initial_positions_dict, py::dict waypoints_dict, std::string &params_filepath)
                      {
            
            // Convert the NumPy array to Eigen MatrixXd
            py::buffer_info buf_info = p_min_npy.request();
            Eigen::Map<Eigen::VectorXd> p_min(reinterpret_cast<double*>(buf_info.ptr),
                                               buf_info.shape[1]);
            buf_info = p_max_npy.request();
            Eigen::Map<Eigen::VectorXd> p_max(reinterpret_cast<double*>(buf_info.ptr),
                                               buf_info.shape[1]);

            std::map<int, Eigen::VectorXd> initial_positions = dictToVectorMap(initial_positions_dict);
            std::map<int, Eigen::MatrixXd> waypoints = dictToMatrixMap(waypoints_dict);

            // Call the constructor with the parameters and waypoints as Eigen MatrixXd
            return new Simulator(num_drones, K, n, delta_t, p_min, p_max, w_g_p, w_g_v, w_s, kappa, v_bar, f_bar, initial_positions, waypoints, params_filepath); }),
             py::arg("num_drones"), py::arg("K"), py::arg("n"), py::arg("delta_t"), py::arg("p_min"), py::arg("p_max"), py::arg("w_g_p"), py::arg("w_g_v"), py::arg("w_s"), py::arg("kappa"), py::arg("v_bar"), py::arg("f_bar"),
             py::arg("initial_positions"), py::arg("waypoints"), py::arg("params_filepath"))
        .def("run_simulation", [](Simulator &instance)
             {
            std::map<int, Eigen::MatrixXd> result = instance.runSimulation();

            // Convert the unordered_map to a Python dict
            py::dict result_dict;
            for (const auto& entry : result) {
                result_dict[py::int_{entry.first}] = entry.second;
            }
            
            return result_dict; });

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
        .def(py::init<std::vector<Drone>, int>())
        .def(py::init([](py::list drone_list, int K) {
            std::vector<Drone> drones;
            for (auto item : drone_list) {
                drones.push_back(item.cast<Drone>());
            }

            return new Swarm(drones, K);
        }))
        .def("solve", [](Swarm &instance, double current_time){
            return instance.solve(current_time);
        });
}