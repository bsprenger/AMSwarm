#include <iostream>
#include <simulator.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(amswarm, m) {
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<int, int, int, float, Eigen::VectorXd, Eigen::VectorXd, float, float, float, int, float, float, Eigen::MatrixXd, Eigen::MatrixXd, std::string&>())
        .def("__init__", [](Simulator& instance, int num_drones, 
                            int K, int n, float delta_t, py::array_t<double> p_min_npy, py::array_t<double> p_max_npy, float w_g_p, float w_g_v, float w_s, int kappa, float v_bar, float f_bar, py::array_t<double> initial_positions_npy, py::array_t<double> waypoints_npy, std::string& params_filepath) {
            
            // Convert the NumPy array to Eigen MatrixXd
            py::buffer_info buf_info = p_min_npy.request();
            Eigen::Map<Eigen::VectorXd> p_min(reinterpret_cast<double*>(buf_info.ptr),
                                               buf_info.shape[1]);
            buf_info = p_max_npy.request();
            Eigen::Map<Eigen::VectorXd> p_max(reinterpret_cast<double*>(buf_info.ptr),
                                               buf_info.shape[1]);

            buf_info = initial_positions_npy.request();
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> initial_positions(reinterpret_cast<double*>(buf_info.ptr),
                                               buf_info.shape[0], buf_info.shape[1]);

            buf_info = waypoints_npy.request();
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> waypoints(reinterpret_cast<double*>(buf_info.ptr),
                                               buf_info.shape[0], buf_info.shape[1]);

            // Call the constructor with the parameters and waypoints as Eigen MatrixXd
            new (&instance) Simulator(num_drones, K, n, delta_t, p_min, p_max, w_g_p, w_g_v, w_s, kappa, v_bar, f_bar, initial_positions, waypoints, params_filepath);
        })
        .def("run_simulation", [](Simulator& instance) {
            Eigen::MatrixXd result = instance.runSimulation();
            Eigen::MatrixXd result_transposed = result.transpose();

            // Convert Eigen matrix to NumPy array
            py::array_t<double> result_array(
                {result.rows(), result.cols()},  // shape
                {result.cols() * sizeof(double), sizeof(double)},  // strides
                result_transposed.data()  // pointer to data
            );
            
            return result_array;
        });
}