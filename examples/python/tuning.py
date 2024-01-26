import amswarm
import numpy as np
import os
import multiprocessing as mp
import sys
from scipy.io import savemat
import itertools

# given a path, iterate through the subfolders and make a big list of all the files
def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def create_new_filename(filename, index=0):
    # Extract the directory path and base filename
    directory, base_filename = os.path.split(filename)
    
    # Extract the last folder name from the directory path
    last_folder = os.path.basename(os.path.normpath(directory))
    
    # Split the base filename into name and extension
    name, extension = os.path.splitext(base_filename)
    
    # Initial filename
    new_filename = f"{last_folder.replace(' ', '_')}_{name}_{index}_results.mat"
    
    # Check if the filename already exists, if yes, increment index until finding a unique filename
    while os.path.exists(os.path.join(directory, new_filename)):
        index += 1
        new_filename = f"{last_folder.replace(' ', '_')}_{name}_{index}_results.mat"
    
    return new_filename

def obj_to_dict(obj):
    return {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('_')}

def load_waypoints(filename):
    data = np.load(filename, allow_pickle=True)
    waypoints = data.item()
    
    initial_positions = {}
    for key in waypoints:
        initial_positions[key] = waypoints[key][0][1:] # first element is time, so skip it
        
    for k, v in waypoints.items():
            waypoints[k] = np.hstack((v, np.zeros((v.shape[0], 3)))) # add zeros for velocity at the waypoints
    return waypoints, initial_positions


def run_swarm_test(filename, config, weights, limits, unique_id):
    waypoints, initial_positions = load_waypoints(filename)
    
    # set up swarm
    amswarm_kwargs = {}
    amswarm_kwargs["config"] = config
    amswarm_kwargs["weights"] = weights
    amswarm_kwargs["limits"] = limits
    amswarm_kwargs["params_filepath"] = "/home/ben/AMSwarm/cpp/params"
    drones = []
    for key in waypoints:
        amswarm_kwargs["waypoints"] = waypoints[key]
        amswarm_kwargs["initial_pos"] = initial_positions[key]
        drones.append(amswarm.Drone(**amswarm_kwargs))
    swarm = amswarm.Swarm(drones)
    
    # results variables
    position_results = []
    control_input_results = []

    # loop variables
    initial_states = []
    prev_inputs = []
    prev_trajectories = []

    for key in waypoints:
        position_results.append(initial_positions[key]) # add initial position to results
        control_input_results.append(np.empty((0,3)))
        initial_states.append(np.concatenate((initial_positions[key], np.zeros(3))))
        prev_inputs.append(np.tile(np.zeros(3), amswarm_kwargs["config"].K))
        prev_trajectories.append(np.tile(initial_positions[key], amswarm_kwargs["config"].K))
        
    opt = [amswarm.SolveOptions()] * len(drones)
    
    step_result = swarm.solve(0.0, initial_states, prev_trajectories, opt, prev_inputs)
    failed_drones = [index for index, drone_result in enumerate(step_result.drone_results) if not drone_result.is_successful]
    if failed_drones:
        # go through solve options for the failed drones and set velocity to false
        for index in failed_drones:
            opt[index].waypoint_velocity_constraints = False
        step_result = swarm.solve(0.0, initial_states, prev_trajectories, opt, prev_inputs)
        
    prev_inputs.clear()
    prev_trajectories.clear()
    for i in range(len(drones)):
        prev_inputs.append(step_result.drone_results[i].control_input_trajectory_vector)
        prev_trajectories.append(step_result.drone_results[i].position_trajectory_vector)
        
    final_waypoint_time = 0.0
    for key in waypoints:
        if waypoints[key][-1,0] > final_waypoint_time:
            final_waypoint_time = waypoints[key][-1,0]

    final_waypoint_time = round(final_waypoint_time / amswarm_kwargs["config"].delta_t) * amswarm_kwargs["config"].delta_t
    num_steps = int(final_waypoint_time / amswarm_kwargs["config"].delta_t)-1 # stop one time step before end -> no control input at last time step
    
    for i in range(num_steps):
        current_time = i * amswarm_kwargs["config"].delta_t
        step_result = swarm.solve(current_time, initial_states, prev_trajectories, opt, prev_inputs)
        failed_drones = [index for index, drone_result in enumerate(step_result.drone_results) if not drone_result.is_successful]
        if failed_drones:
            for index in failed_drones:
                opt[index].waypoint_velocity_constraints = False
            step_result = swarm.solve(current_time, initial_states, prev_trajectories, opt, prev_inputs)
            
        initial_states.clear()
        prev_inputs.clear()
        prev_trajectories.clear()
        for drone in range(len(drones)):
            initial_states.append(step_result.drone_results[drone].state_trajectory[0,:])
            
            # Get the last two control inputs
            last_input = step_result.drone_results[drone].control_input_trajectory[-1,:]
            second_last_input = step_result.drone_results[drone].control_input_trajectory[-2,:]
            extrapolated_input = 2 * last_input - second_last_input
            
            new_inputs = np.hstack((step_result.drone_results[drone].control_input_trajectory_vector[3:], extrapolated_input))
            prev_inputs.append(new_inputs)
            
            # Get the last two points of the trajectory
            last_point = step_result.drone_results[drone].position_trajectory[-1][:3]  # Only take x, y, z coordinates
            second_last_point = step_result.drone_results[drone].position_trajectory[-2][:3]  # Only take x, y, z coordinates
            extrapolated_point = 2 * last_point - second_last_point

            new_trajectory = np.hstack((step_result.drone_results[drone].position_trajectory_vector[3:], extrapolated_point))
            prev_trajectories.append(new_trajectory)
            
            # prev_trajectories.append(step_result.drone_results[i].position_trajectory_vector)
            
            # Here we also keep track of our position and control input for plotting later
            position_results[drone] = np.vstack((position_results[drone], step_result.drone_results[drone].position_trajectory[0,:]))
            control_input_results[drone] = np.vstack((control_input_results[drone], step_result.drone_results[drone].control_input_trajectory[0,:]))
    
    # save to file
    config_dict = obj_to_dict(config)
    weights_dict = obj_to_dict(weights)
    limits_dict = obj_to_dict(limits)
    string_key_waypoints = {f'key_{k}': v for k, v in waypoints.items()}
    
    save_filename = create_new_filename(filename, index=unique_id)
    print(save_filename)
    mat_data = {
        'waypoints': string_key_waypoints,
        'position_results': position_results,
        'control_input_results': control_input_results,
        'config': config_dict,
        'weights': weights_dict,
        'limits': limits_dict,
        'filename': filename
    }
    savemat(save_filename, mat_data)
    

def main(folder):
    # get list of all files
    file_list = get_file_list(folder)
    
    # remove all non-npy files
    file_list = [x for x in file_list if x.endswith(".npy")]
    
    # take only the first 2 files for testing
    file_list = file_list[:1]
    
    # weights
    weights = []
    weights.append(amswarm.MPCWeights())
    w = amswarm.MPCWeights()
    w.w_goal_vel = 10001
    weights.append(w)
    print(weights[0].w_goal_vel)
    
    # configs
    configs = []
    configs.append(amswarm.MPCConfig())
    
    # limits
    limits = []
    limits.append(amswarm.PhysicalLimits())
    
    # generate all permutations of file_list, configs, weights, and limits
    permutations = itertools.product(file_list, configs, weights, limits)
    pool = mp.Pool(mp.cpu_count())
    for idx, perm in enumerate(permutations):
        pool.apply_async(run_swarm_test, args=(*perm, idx))
    pool.close()
    pool.join()
    # # now, run swarm test on each combination IN PARALLEL WITH MULTIPROCESSING
    
    # pool.starmap(run_swarm_test, permutations)

    


if __name__ == '__main__':
    folder = sys.argv[1]
    main(folder)