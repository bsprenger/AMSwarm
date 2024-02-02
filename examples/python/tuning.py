import amswarm
import numpy as np
import os
import multiprocessing as mp
import sys
from scipy.io import savemat
import itertools
import time
import yaml

# given a path, iterate through the subfolders and make a big list of all the files
def get_file_list(path):
    return [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]


def create_new_filename(filename, index=0):
    directory, base_filename = os.path.split(filename)
    last_folder = os.path.basename(os.path.normpath(directory)).replace(' ', '_')
    name, extension = os.path.splitext(base_filename)
    new_filename = f"{last_folder}_{name}_{index}_results.mat"
    
    while os.path.exists(os.path.join(directory, new_filename)):
        index += 1
        new_filename = f"{last_folder}_{name}_{index}_results.mat"
    
    return new_filename


def obj_to_dict(obj):
    return {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('_')}


def load_waypoints(filename):
    data = np.load(filename, allow_pickle=True).item()
    initial_positions = {key: value[0][1:] for key, value in data.items()}
    waypoints = {k: np.hstack((v, np.zeros((v.shape[0], 3)))) for k, v in data.items()}
    return waypoints, initial_positions


def setup_amswarm_kwargs(config, weights, limits):
    amswarm_kwargs = {
        "config": config,
        "weights": weights,
        "limits": limits,
        "dynamics": amswarm.SparseDynamics()
    }
    
    with open(os.path.join(os.getcwd(), 'cpp', 'params', 'model_params.yaml'), 'r') as file:
        data = yaml.safe_load(file)
        
    for key in ['A', 'B', 'A_prime', 'B_prime']:
        setattr(amswarm_kwargs["dynamics"], key, np.array(data['dynamics'][key]))

    return amswarm_kwargs


def disable_constraints_for_failed_drones(opt, failed_drones):
    for index in failed_drones:
        opt[index].waypoint_position_constraints = False
        opt[index].waypoint_velocity_constraints = False
        opt[index].waypoint_acceleration_constraints = False
        

def disable_continuity_constraints(opt):
    for options in opt:
        options.input_continuity_constraints = False
        options.input_dot_continuity_constraints = False
        options.input_ddot_continuity_constraints = False


def enable_continuity_constraints(opt):
    for options in opt:
        options.input_continuity_constraints = True
        options.input_dot_continuity_constraints = True
        options.input_ddot_continuity_constraints = True
        
        
def solve_swarm(swarm, current_time, initial_states, prev_trajectories, opt, prev_inputs):
    num_drones = len(prev_inputs)
    # reset constraints to be ON
    for drone in range(num_drones):
        opt[drone].waypoint_position_constraints = True
        opt[drone].waypoint_velocity_constraints = True
        opt[drone].waypoint_acceleration_constraints = False
    step_result = swarm.solve(current_time, initial_states, prev_trajectories, opt, prev_inputs)
    failed_drones = [index for index, drone_result in enumerate(step_result.drone_results) if not drone_result.is_successful]
    if failed_drones:
        disable_constraints_for_failed_drones(opt, failed_drones)
        step_result = swarm.solve(current_time, initial_states, prev_trajectories, opt, prev_inputs)
    return step_result
        

def run_swarm_test(filename, config, weights, limits, unique_id):
    # Setup swarm
    waypoints, initial_positions = load_waypoints(filename)
    amswarm_kwargs = setup_amswarm_kwargs(config, weights, limits)
    num_inputs = amswarm_kwargs["dynamics"].B.shape[1]
    drones = [amswarm.Drone(waypoints=waypoints[key], initial_pos=initial_positions[key], **amswarm_kwargs) for key in waypoints]
    swarm = amswarm.Swarm(drones)
    
    # Create results variables
    position_results = [initial_positions[key] for key in waypoints]
    control_input_results = [np.empty((0, num_inputs)) for _ in waypoints]

    # Create loop variables
    initial_states = []
    prev_inputs = []
    prev_trajectories = []

    for key in waypoints:
        initial_states.append(np.concatenate((initial_positions[key], np.zeros(3))))
        prev_inputs.append(np.tile(np.zeros(num_inputs), amswarm_kwargs["config"].K))
        prev_trajectories.append(np.tile(initial_positions[key], amswarm_kwargs["config"].K))
        
    opt = [amswarm.SolveOptions() for _ in drones]
    disable_continuity_constraints(opt) # for the first step, we don't need continuity constraints
    
    # start timer
    start = time.time()
    
    step_result = solve_swarm(swarm, 0.0, initial_states, prev_trajectories, opt, prev_inputs)
        
    # enable_continuity_constraints(opt) # for the rest of the steps, we need continuity constraints
        
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
        
        step_result = solve_swarm(swarm, current_time, initial_states, prev_trajectories, opt, prev_inputs)
            
        initial_states.clear()
        prev_inputs.clear()
        prev_trajectories.clear()
        for drone in range(len(drones)):
            initial_states.append(step_result.drone_results[drone].state_trajectory[0,:])
            
            # Get the last two control inputs
            last_input = step_result.drone_results[drone].control_input_trajectory[-1,:]
            second_last_input = step_result.drone_results[drone].control_input_trajectory[-2,:]
            extrapolated_input = 2 * last_input - second_last_input
            
            new_inputs = np.hstack((step_result.drone_results[drone].control_input_trajectory_vector[num_inputs:], extrapolated_input))
            prev_inputs.append(new_inputs)
            
            # Get the last two points of the trajectory
            last_point = step_result.drone_results[drone].position_trajectory[-1][:3]  # Only take x, y, z coordinates
            second_last_point = step_result.drone_results[drone].position_trajectory[-2][:3]  # Only take x, y, z coordinates
            extrapolated_point = 2 * last_point - second_last_point

            new_trajectory = np.hstack((step_result.drone_results[drone].position_trajectory_vector[3:], extrapolated_point))
            prev_trajectories.append(new_trajectory)
            
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
        'filename': filename,
        'time_elapsed': time.time() - start
    }
    savemat(save_filename, mat_data)
    

def main(folder):
    file_list = [x for x in get_file_list(folder) if x.endswith(".npy")]
    file_list = file_list[:4]  # Limiting files for testing
    
    
    # weights
    weights = []
    w1 = amswarm.MPCWeights()
    w1.w_input_continuity = 0.0
    w1.w_input_dot_continuity = 0.0
    w1.w_input_ddot_continuity = 0.0
    w2 = amswarm.MPCWeights()
    w2.w_input_continuity = 1000.0
    w2.w_input_dot_continuity = 1000.0
    w2.w_input_ddot_continuity = 1000.0
    w3 = amswarm.MPCWeights()
    w3.w_input_continuity = 10000.0
    w3.w_input_dot_continuity = 10000.0
    w3.w_input_ddot_continuity = 10000.0
    w4 = amswarm.MPCWeights()
    w4.w_input_continuity = 100000.0
    w4.w_input_dot_continuity = 100000.0
    w4.w_input_ddot_continuity = 100000.0
    weights.append(w1)
    weights.append(w2)
    weights.append(w3)
    weights.append(w4)
    
    
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


if __name__ == '__main__':
    folder = sys.argv[1]
    main(folder)