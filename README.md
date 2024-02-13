# AMSwarm
New version of AMSwarm for Swarm GPT. 

ROUGH NOTES FOR NOW, to be fixed later

# Build instructions

You can either build just the C++ library by itself or both the CPP and the Python binding.

## Building C++ and Python binding

In the project's main folder, run

```
:$ pip install .
```

You should now be able to import amswarm in Python as

```
import amswarm
```


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


