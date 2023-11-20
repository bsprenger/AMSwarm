// python/example.cpp
#include <iostream>
#include <simulator.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// void helloWorld() {
//     std::cout << "Hello, World!" << std::endl;
// }

// PYBIND11_MODULE(amswarm, m) {
//     m.def("hello_world", &helloWorld);
// }

PYBIND11_MODULE(amswarm, m) {
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<>());
}