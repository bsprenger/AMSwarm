import amswarm
import numpy as np

# Define initial positions and waypoints
initial_positions = {72: np.array([1,0,1]), 1: np.array([0,1,1]), 2: np.array([0,-1,1])}
waypoints = {72: np.array([[2.25, 1, 0, 1, 0, 0, 0],
                    [4.13, 1.3, 0.7, 1.5, 0, 0, 0],
                    [6.04, 1, 1, 2, 0, 0, 0],
                    [7.92, 0.7, 0.7, 2.5, 0, 0, 0],
                    [9.82, 0, 1, 3, 0, 0, 0],
                    [11.7, -0.7, 0.7, 2.5, 0, 0, 0],
                    [13.61, -1, 1, 2, 0, 0, 0],
                    [15.51, -0.7, 0.7, 1.5, 0, 0, 0],
                    [17.41, -1, 0, 1, 0, 0, 0],
                    [19.3, -0.7, -0.7, 1.5, 0, 0, 0],
                    [21.2, 0, -1, 2, 0, 0, 0],
                    [23.08, 0.7, -0.7, 2.5, 0, 0, 0],
                    [24.98, 1, -1, 3, 0, 0, 0],
                    [26.89, 1.3, -0.7, 2.5, 0, 0, 0],
                    [28.79, 1, 0, 2, 0, 0, 0],
                    [30.67, 1, 0, 1, 0, 0, 0]]),
            1: np.array([[2.25, 0, 1, 1, 0, 0, 0],
                    [4.13, -0.7, 1.3, 1.5, 0, 0, 0],
                    [6.04, -1, 1, 2, 0, 0, 0],
                    [7.92, -0.7, 0.7, 2.5, 0, 0, 0],
                    [9.82, -1, 0, 3, 0, 0, 0],
                    [11.7, -0.7, -0.7, 2.5, 0, 0, 0],
                    [13.61, -1, -1, 2, 0, 0, 0],
                    [15.51, -0.7, -0.7, 1.5, 0, 0, 0],
                    [17.41, 0, -1, 1, 0, 0, 0],
                    [19.3, 0.7, -0.7, 1.5, 0, 0, 0],
                    [21.2, 1, -1, 2, 0, 0, 0],
                    [23.08, 1.3, -0.7, 2.5, 0, 0, 0],
                    [24.98, 1, 0, 3, 0, 0, 0],
                    [26.89, 0.7, 0.7, 2.5, 0, 0, 0],
                    [28.79, 1, 1, 2, 0, 0, 0],
                    [30.67, 1, 1, 1, 0, 0, 0]]),
            2: np.array([[2.25, 0, -1, 1, 0, 0, 0],
                    [4.13, 0.7, -1.3, 1.5, 0, 0, 0],
                    [6.04, 1, -1, 2, 0, 0, 0],
                    [7.92, 0.7, -0.7, 2.5, 0, 0, 0],
                    [9.82, 1, 0, 3, 0, 0, 0],
                    [11.7, 0.7, 0.7, 2.5, 0, 0, 0],
                    [13.61, 1, 1, 2, 0, 0, 0],
                    [15.51, 0.7, 0.7, 1.5, 0, 0, 0],
                    [17.41, 0, 1, 1, 0, 0, 0],
                    [19.3, -0.7, 0.7, 1.5, 0, 0, 0],
                    [21.2, -1, 1, 2, 0, 0, 0],
                    [23.08, -1.3, 0.7, 2.5, 0, 0, 0],
                    [24.98, -1, 0, 3, 0, 0, 0],
                    [26.89, -0.7, -0.7, 2.5, 0, 0, 0],
                    [28.79, -1, -1, 2, 0, 0, 0],
                    [30.67, -1, -1, 1, 0, 0, 0]])}


# Define params that are constant for all drones
amswarm_kwargs = {}
amswarm_kwargs["delta_t"] = 1/6
amswarm_kwargs["K"] = 25
amswarm_kwargs["n"] = 10
amswarm_kwargs["p_min"] = np.array([[-10,-10,0]])
amswarm_kwargs["p_max"] = np.array([[10, 10, 10]])
amswarm_kwargs["w_g_p"] = 7000
amswarm_kwargs["w_g_v"] = 1000
amswarm_kwargs["w_s"] = 100
amswarm_kwargs["v_bar"] = 1.73
amswarm_kwargs["f_bar"] = 1.5 * 9.81
amswarm_kwargs["params_filepath"] = "/home/ben/AMSwarm/cpp/params"
amswarm_kwargs["hard_waypoint_constraints"] = False
amswarm_kwargs["acceleration_constraints"] = False


## --------------------------- CREATE SWARM -------------------------------- ##
# First, create a list of individual drones. AMSwarm does not consider drone
# IDs (as that it irrelevant to the optimization problem). Therefore we create
# a new drone for each key in the waypoints/initial positions dictionaries.
# Each drone is created individually such that they have their own set of
# waypoints, initial positions, and parameters.
drones = []
for key in waypoints:
    amswarm_kwargs["waypoints"] = waypoints[key]
    amswarm_kwargs["initial_pos"] = initial_positions[key]
    print(amswarm_kwargs)
    drones.append(amswarm.Drone(**amswarm_kwargs))

# Then, we create a swarm controller by passing our list of drones to it,
# While each drone is created separately and can have their own parameters,
# there are a few parameters that they need to share in common in order for the
# swarm controller to be able to coordinate them. These parameters are:
#   - delta_t
#   - K
# Each drone needs to know all other drones' positions at each time step in its
# horizon. Therefore, the timestep and the horizon length must be the same for
# all drones. Once our list of drones are created, we create the swarm:
swarm = amswarm.Swarm(drones)


## -------------------- SIMULATE ENTIRE TRAJECTORY ------------------------- ##
# Now that our swarm has been created, we can simulate the entire trajectory
# from time 0.0 to the final waypoint time. We do this if we do not care about
# the intermediate trajectories that the optimization outputs, and only want
# the final result. Essentially, this method assumes that the dynamics model
# that we are using is perfect, so the drone moves perfectly to the next
# predicted state in its horizon. This method replcaes the need for some
# external simulator (like gym-pybullet-drones) to simulate where the drone
# moves to next given an input. However, obviously it will not be as accurate
# to reality as a real simulator. It is useful as a first check to see if the
# drone is likely to be able to reach the waypoints at all. If it is not able
# to reach the waypoints with the assumption of a perfect model, then it will 
# almost certainly not be able to if we run the optimizer in closed-loop with 
# a higher-fidelity simulator like gym-pybullet-drones.

sim_result = swarm.run_simulation()

# The output sim_result is a class/struct with one main parameter that we use:
#   - drone_results -> a list of DroneResult classes/structs, one for each drone

# Each DroneResult object in the drone_results vector has the following:
#   - position_trajectory -> predicted path that it will follow (x,y,z)
#   - state_trajectory -> predicted path AND velocity (x,y,z,vx,vy,vz)
#   - control_input_trajectory -> what you would actually apply to the system
#       - currently these are x,y,z position references
#   - position_state_time_stamps -> time stamps for the above trajectories
#   - control_input_time_stamps -> time stamps for the control inputs

# For example, to print the predicted trajectory of the first drone:
# print(sim_result.drone_results[0].position_trajectory)

# To print the control inputs that would be applied to the first drone:
# print(sim_result.drone_results[0].control_input_trajectory)


## ------------------- SOLVE FOR TRAJECTORY IN REAL TIME ------------------- ##
# Now we will solve for the trajectory in real time. This is the main use case
# for AMSwarm. We will solve for the trajectory and control inputs at each time
# step, apply the inputs to the drones, and then measure the next state. Then,
# we will solve for the next time step starting at the new initial state, and 
# so on. This is the method that we would use if we were running the optimizer
# in closed-loop with a higher-fidelity simulator like gym-pybullet-drones, or
# running in real life on a real swarm.

# First, we need to create a list of initial states and initial trajectory
# guesses. EAch drone needs a guess for the other drones' trajectories so that
# it can avoid collisions. When we start, we have no guess for the other drones'
# trajectories. To solve this, we run an initial optimization with each drone
# assuming the other drones are stationary, so that they plan an unobstructed
# path to their first waypoint targets. This is not a perfect solution, but it
# is a good enough guess to get the drones moving. We will use the initial
# positions as the initial states, and we will use the initial positions as the
# initial trajectory guesses.
# We also initialize the results lists to store the results from each time step
position_results = []
control_input_results = []

initial_states = []
prev_trajectories = []

for key in waypoints:
    position_results.append(initial_positions[key]) # add initial position to results
    control_input_results.append(np.empty((0,3)))
    initial_states.append(np.concatenate((initial_positions[key], np.zeros(3))))
    prev_trajectories.append(np.tile(initial_positions[key], amswarm_kwargs["K"]))

# Get our initial guesses for the drone trajectories
step_result = swarm.solve(0.0, initial_states, prev_trajectories)
prev_trajectories.clear()
for i in range(len(drones)):
    prev_trajectories.append(step_result.drone_results[i].position_trajectory_vector)

# Now we have good initial trajectory guesses, we can solve the optimization
# at each time step and apply the control inputs to the drones.
# First, we find the final waypoint time and round to the nearest multiple of
# delta_t. Then, we find the number of steps that we need to take to get to the
# final waypoint time. We will stop one time step before the final waypoint time
# because the last control input should occur at the second-to-last time step.
final_waypoint_time = 0.0
for key in waypoints:
    if waypoints[key][-1,0] > final_waypoint_time:
        final_waypoint_time = waypoints[key][-1,0]
        
final_waypoint_time = round(final_waypoint_time / amswarm_kwargs["delta_t"]) * amswarm_kwargs["delta_t"]
num_steps = int(final_waypoint_time / amswarm_kwargs["delta_t"])-1 # stop one time step before end -> no control input at last time step

# Now we can solve for the trajectory at each time step, apply it to the drones,
# measure the new state, and repeat.
for i in range(num_steps):
    
    # Solve for the control input given the current state and guesses for the 
    # other drones' trajectories from the previous optimization
    current_time = i * amswarm_kwargs["delta_t"]
    step_result = swarm.solve(current_time, initial_states, prev_trajectories)
    
    # Here, we would apply the control inputs to the drones in real life or in
    # a simulator. For now, we will just print the control inputs that would be
    # applied to the drones.
    # print("Control Inputs at Time " + str(current_time) + ":")
    # for j in range(len(drones)):
    #     print("Drone " + str(j) + ": " + str(step_result.drone_results[j].control_input_trajectory[0,:]))

    # We get the new guesses for the drone trajectories from the optimization.
    # Here we also take the next predicted state as the initial state for
    # the next optimization (effectively assuming a perfect model). This would
    # be replaced with a measurement of the initial state in real life or in a
    # simulator (if measuring, would need to wait until the next time step).
    initial_states.clear()
    prev_trajectories.clear()
    for i in range(len(drones)):
        initial_states.append(step_result.drone_results[i].state_trajectory[0,:])
        prev_trajectories.append(step_result.drone_results[i].position_trajectory_vector)
        
        # Here we also keep track of our position and control input for plotting later
        position_results[i] = np.vstack((position_results[i], step_result.drone_results[i].position_trajectory[0,:]))
        control_input_results[i] = np.vstack((control_input_results[i], step_result.drone_results[i].control_input_trajectory[0,:]))
        
        
# We now have simulated our drones' trajectories from time 0 until the final
# waypoint time. We can inspect the results for the 1st drone:
print(position_results[0])
print(control_input_results[0])