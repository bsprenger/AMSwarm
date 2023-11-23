#include <iostream>
#include <simulator.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::unordered_map<int, Eigen::MatrixXd> dictToMatrixMap(py::dict d)
{
    std::unordered_map<int, Eigen::MatrixXd> result;

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

std::unordered_map<int, Eigen::VectorXd> dictToVectorMap(py::dict d)
{
    std::unordered_map<int, Eigen::VectorXd> result;

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
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<int, int, int, float, Eigen::VectorXd, Eigen::VectorXd, float, float, float, int, float, float, std::unordered_map<int, Eigen::VectorXd>, std::unordered_map<int, Eigen::MatrixXd>, std::string &>())
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

            std::unordered_map<int, Eigen::VectorXd> initial_positions = dictToVectorMap(initial_positions_dict);
            std::unordered_map<int, Eigen::MatrixXd> waypoints = dictToMatrixMap(waypoints_dict);

            // Call the constructor with the parameters and waypoints as Eigen MatrixXd
            return new Simulator(num_drones, K, n, delta_t, p_min, p_max, w_g_p, w_g_v, w_s, kappa, v_bar, f_bar, initial_positions, waypoints, params_filepath); }),
             py::arg("num_drones"), py::arg("K"), py::arg("n"), py::arg("delta_t"), py::arg("p_min"), py::arg("p_max"), py::arg("w_g_p"), py::arg("w_g_v"), py::arg("w_s"), py::arg("kappa"), py::arg("v_bar"), py::arg("f_bar"),
             py::arg("initial_positions"), py::arg("waypoints"), py::arg("params_filepath"))
        .def("run_simulation", [](Simulator &instance)
             {
            std::unordered_map<int, Eigen::MatrixXd> result = instance.runSimulation();

            // Convert the unordered_map to a Python dict
            py::dict result_dict;
            for (const auto& entry : result) {
                result_dict[py::int_{entry.first}] = entry.second;
            }
            
            return result_dict; });
}