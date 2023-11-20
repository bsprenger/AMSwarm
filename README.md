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

sim = amswarm.Simulator()
```


