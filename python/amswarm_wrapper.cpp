#include <iostream>
#include <simulator.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(amswarm, m) {
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<const Eigen::MatrixXd&>())
        .def("__init__", [](Simulator& instance, py::array_t<double> input_array) {
            // Convert the NumPy array to Eigen MatrixXd
            py::buffer_info buf_info = input_array.request();
            Eigen::Map<Eigen::MatrixXd> matrix(reinterpret_cast<double*>(buf_info.ptr),
                                               buf_info.shape[0], buf_info.shape[1]);

            // Call the constructor with the Eigen MatrixXd
            new (&instance) Simulator(matrix);
        })
        .def("run_simulation", [](Simulator& instance) {
            Eigen::MatrixXd result = instance.runSimulation();
            
            // Convert Eigen matrix to NumPy array
            py::array_t<double> result_array({result.rows(), result.cols()}, result.data());
            
            return result_array;
        });
}