#include <pybind11/pybind11.h>
#include "simulator.h"

namespace py = pybind11;

PYBIND11_MODULE(amswarm, m) {
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<const std::string&>())
        .def("run", &Simulator::run)
        .def_readwrite("num_drones", &Simulator::num_drones)
        .def_readwrite("initial_positions", &Simulator::initial_positions)
        .def_readwrite("waypoints", &Simulator::waypoints)
        .def_readwrite("K", &Simulator::K)
        .def_readwrite("n", &Simulator::n)
        .def_readwrite("delta_t", &Simulator::delta_t)
        .def_readwrite("p_min", &Simulator::p_min)
        .def_readwrite("p_max", &Simulator::p_max)
        .def_readwrite("w_g_p", &Simulator::w_g_p)
        .def_readwrite("w_g_v", &Simulator::w_g_v)
        .def_readwrite("w_s", &Simulator::w_s)
        .def_readwrite("kappa", &Simulator::kappa)
        .def_readwrite("v_bar", &Simulator::v_bar)
        .def_readwrite("f_bar", &Simulator::f_bar)
        .def_readwrite("swarm", &Simulator::swarm);
}