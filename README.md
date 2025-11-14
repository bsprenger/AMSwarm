# AMSwarm 2.0

[![CI](https://github.com/bsprenger/AMSwarm/actions/workflows/ci.yml/badge.svg)](https://github.com/bsprenger/AMSwarm/actions/workflows/ci.yml)

## Overview
AMSwarm 2.0 is a high-speed drone swarm trajectory planner that improves upon the original [AMSwarm](https://github.com/utiasDSL/AMSwarm) implementation (and associated [paper](https://arxiv.org/abs/2303.04856)).

The core of AMSwarm 2.0 is written in C++, using the Eigen linear algebra library for high-speed, real-time suitable code. We also provide an easy to use Python wrapper that allows for seamless integration to existing simulators such as [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones).

## Key Enhancements

- **Temporal Waypoint Tracking**: Targets precise arrival times at waypoints, optimizing for temporal accuracy.
- **Dynamics Model Incorporation**: Uses identified dynamics models for enhanced trajectory prediction and collision avoidance.
- **Optimized AM Algorithm**: Improves the Alternating-Minimization algorithm for greater efficiency, completely eliminating slow trigonometric operations.
- **Distributed Avoidance Responsibilities**: Distributes avoidance tasks among drones to reduce collision constraints by half.
- **Improved Code Structure**: Abstracts the AM algorithm for easier adaptation and customization to various use cases.

# Build instructions

## Prerequisites

Tested on Ubuntu 20.04 with Python 3.8. Requirements include:

- CMake (>=3.14)
- C++ compiler with OpenMP support
- Ninja build system (recommended)
- [pybind11](https://pybind11.readthedocs.io/en/stable/installing.html) (Note: depending on your system, you may need to install the pybind11-global option for CMake visibility)

## Installing AMSwarm
1. Clone the AMSwarm repository:
```
git clone https://github.com/bsprenger/AMSwarm.git
```
2. Navigate to the cloned directory and install with pip:
```
pip install .
```
This will automatically compile the C++ code and install the Python module in the appropriate location for your Python installation.

## Building for Development

This project uses CMake Presets for standardized build configurations. See [CMAKE_PRESETS.md](CMAKE_PRESETS.md) for detailed instructions.

Quick start:
```bash
# Debug build
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

# Release build
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Available presets include: debug, release, relwithdebinfo, asan, tsan, ubsan, clang-tidy, coverage, gcc-release, and clang-release.

# Using AMSwarm

You can use AMSwarm in C++ by including the appropriate headers in your project and linking against the compiled library. For Python usage, after installation, import the `amswarm` module in your scripts:

```
import amswarm
```

Refer to the examples provided in the `examples/python` directory for comprehensive usage scenarios.

# Contributing

We welcome contributions to AMSwarm! If you have suggestions for improvements or encounter any issues, please open an issue or pull request on our GitHub repository.

# License

AMSwarm is released under the MIT License. See the LICENSE file for more details.

# References

[1] Vivek K. Adajania, Siqi Zhou, Arun Kumar Singh, and Angela P. Schoellig. Amswarm: An alternating minimization approach for safe motion planning of quadrotor swarms in cluttered environments. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 1421–1427, 2023.

[2] [Original AMSwarm repository](https://github.com/utiasDSL/AMSwarm)