# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ben/AMSwarm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ben/AMSwarm/build

# Include any dependencies generated for this target.
include CMakeFiles/amswarm_py.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/amswarm_py.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/amswarm_py.dir/flags.make

CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.o: CMakeFiles/amswarm_py.dir/flags.make
CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.o: ../python/simulator_py.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/AMSwarm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.o -c /home/ben/AMSwarm/python/simulator_py.cpp

CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/AMSwarm/python/simulator_py.cpp > CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.i

CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/AMSwarm/python/simulator_py.cpp -o CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.s

# Object files for target amswarm_py
amswarm_py_OBJECTS = \
"CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.o"

# External object files for target amswarm_py
amswarm_py_EXTERNAL_OBJECTS =

amswarm_py.cpython-38-x86_64-linux-gnu.so: CMakeFiles/amswarm_py.dir/python/simulator_py.cpp.o
amswarm_py.cpython-38-x86_64-linux-gnu.so: CMakeFiles/amswarm_py.dir/build.make
amswarm_py.cpython-38-x86_64-linux-gnu.so: libAMSwarm.so
amswarm_py.cpython-38-x86_64-linux-gnu.so: CMakeFiles/amswarm_py.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ben/AMSwarm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module amswarm_py.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/amswarm_py.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/ben/AMSwarm/build/amswarm_py.cpython-38-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/amswarm_py.dir/build: amswarm_py.cpython-38-x86_64-linux-gnu.so

.PHONY : CMakeFiles/amswarm_py.dir/build

CMakeFiles/amswarm_py.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/amswarm_py.dir/cmake_clean.cmake
.PHONY : CMakeFiles/amswarm_py.dir/clean

CMakeFiles/amswarm_py.dir/depend:
	cd /home/ben/AMSwarm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ben/AMSwarm /home/ben/AMSwarm /home/ben/AMSwarm/build /home/ben/AMSwarm/build /home/ben/AMSwarm/build/CMakeFiles/amswarm_py.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/amswarm_py.dir/depend

