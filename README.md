# AMSwarm
New version of AMSwarm for Swarm GPT. 

ROUGH NOTES FOR NOW, to be fixed later

# Build instructions

You can either build just the C++ library by itself or both the CPP and the Python binding.

## Building C++ and Python binding

You will need yaml-cpp and pybind11 installed. Python 3.8 has been tested and works, unsure about other versions.

Inside the main AMSwarm folder:


```
mkdir build
cd build
cmake ..
make
make install
```

This will install the files for the AMSwarm C++ library to `/home/{user}/.local`. If you get import errors when using the AMSwarm library in python (e.g. `cannot find libAMSwarm.so`) then you may need to add this location to your linker path:


```
export LD_LIBRARY_PATH=/home/{your user name}/.local/lib:${LD_LIBRARY_PATH}
```

TO DO: This needs to be done in every terminal so that the python library can find the AMSwarm library. this needs to be changed at some point so that the files for AMSwarm are in a better location.

The python library is installed to the same folder where pybind11 was installed. This is up for change, I did this at the moment because I used pip to install pybind so I knew that python would be able to find the amswarm python library.

# Using the library

Currently AMSwarm is accessible through the Simulator class which takes in waypoints for all drones, creates a Swarm, and runs the optimization until they have reached all the waypoints (or stops if some waypoints are unreachable or other errors occur). Other interfaces will be created in future to directly control the swarm or individual drones.

Usage in python:

```
import amswarm
import numpy as np

# for now, waypoint timing MUST BE IN ORDER - will give bad results if time steps are not in order
# if one drone has no waypoints remaining while others do, it will drift - make sure that all drones have a waypoint at final position to avoid this.
# first index: drone ID from 0 to num_drones - 1
# second index: time
# remaining indices: xyz position, xyz velocity
waypoints = np.array([[0, 1, 1, 1, 1, 0,0,0], [1, 1, 2, 2, 2, 0, 0, 0], [2, 1, 3, 3, 3, 0, 0, 0]])

num_drones = 3
K = 15
n = 10
delta_t = 0.2
p_min = np.array([[-10,-10,-10]])
p_max = np.array([[10, 10, 10]])
w_g_p = 7000
w_g_v = 0
w_s = 0
kappa = 1
v_bar = 1
f_bar = 10

initial_positions = np.array([[0,9,9,9],[1,1,2,3],[2,2,3,4]])

params_filepath = "/home/ben/AMSwarm/cpp/params"


sim = amswarm.Simulator(num_drones, K, n, delta_t, p_min, p_max, w_g_p, w_g_v, w_s, kappa, v_bar, f_bar, initial_positions, waypoints, params_filepath)

results = sim.run_simulation()
print(results)
```


