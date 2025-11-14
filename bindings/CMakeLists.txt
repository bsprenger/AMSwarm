# Python bindings for AMSwarm
if(NOT BUILD_PYTHON_BINDINGS)
    return()
endif()

# Find pybind11
find_package(pybind11 REQUIRED) 

# Python wrapper source files
set(PYTHON_WRAPPER_FILES
    amswarm_wrapper.cpp
)

# Create Python module
pybind11_add_module(amswarm MODULE ${PYTHON_WRAPPER_FILES})

# Link the C++ library
target_link_libraries(amswarm PRIVATE AMSwarm::AMSwarm)
