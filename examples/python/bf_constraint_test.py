import amswarm
import numpy as np
import yaml
import os
from pathlib import Path

def load_yaml_file(file_path: str) -> dict:
    """Load YAML configuration file."""
    with open(Path(__file__).resolve().parent / file_path, "r") as f:
        return yaml.safe_load(f)
    

def initialize_swarm_optimization(waypoints: dict, settings: dict) -> tuple:
    """Initialize swarm optimization with initial guesses."""
    initial_states, prev_inputs, prev_trajectories = [], [], []
    for key, waypoint in waypoints.items():
        initial_pos = waypoint[0, 1:4]  # Assuming waypoint format [time, x, y, z]
        initial_states.append(np.concatenate((initial_pos, np.zeros(3))))
        prev_inputs.append(
            np.tile(np.concatenate((initial_pos, np.zeros(3))), settings["MPCConfig"]["K"]))
        prev_trajectories.append(np.tile(initial_pos, settings["MPCConfig"]["K"]))
    return initial_states, prev_inputs, prev_trajectories


def disable_continuity_constraints(opt: amswarm.SolveOptions):
    """Disable continuity constraints for the optimization."""
    for options in opt:
        options.input_continuity_constraints = False
        options.input_dot_continuity_constraints = False
        options.input_ddot_continuity_constraints = False


def enable_continuity_constraints(opt: amswarm.SolveOptions):
    """Enable continuity constraints for the optimization."""
    for options in opt:
        options.input_continuity_constraints = True
        options.input_dot_continuity_constraints = True
        options.input_ddot_continuity_constraints = True


def disable_constraints_for_failed_drones(opt: amswarm.SolveOptions, failed_drones):
    """Disable constraints for drones that failed to solve the optimization."""
    for index in failed_drones:
        opt[index].waypoint_position_constraints = False
        opt[index].waypoint_velocity_constraints = False
        opt[index].waypoint_acceleration_constraints = False


def reset_constraints(opt: amswarm.SolveOptions):
    """Reset constraints for the optimization."""
    for options in opt:
        options.waypoint_position_constraints = True
        options.waypoint_velocity_constraints = True
        options.waypoint_acceleration_constraints = False

    
def solve_swarm(swarm: amswarm.Swarm, current_time: float, initial_states: np.array,
                prev_trajectories: np.array, opt: amswarm.SolveOptions,
                prev_inputs: np.array) -> amswarm.SwarmResult:
    """Solve the swarm optimization problem. If it fails, disable constraints and try again."""
    step_result = swarm.solve(current_time, initial_states, prev_trajectories, opt, prev_inputs)
    failed_drones = [
        index for index, drone_result in enumerate(step_result.drone_results)
        if not drone_result.is_successful
    ]
    if failed_drones:
        disable_constraints_for_failed_drones(opt, failed_drones)
        step_result = swarm.solve(current_time, initial_states, prev_trajectories, opt, prev_inputs)
        reset_constraints(opt)
    return step_result


def get_last_two_elements(array: np.array):
    """Return the last two elements of an array."""
    return array[-1, :], array[-2, :]


def extrapolate(last_element: np.array, second_last_element: np.array) -> np.array:
    """Extrapolate the next element based on the last two elements."""
    return 2 * last_element - second_last_element


def update_for_next_iteration(step_result: amswarm.SwarmResult):
    """Update states, inputs, and trajectories for the next iteration."""
    initial_states = []
    prev_inputs = []
    prev_trajectories = []
    
    for drone_index, result in enumerate(step_result.drone_results):
        initial_states.append(result.state_trajectory[0,:])
        last_input, second_last_input = get_last_two_elements(result.control_input_trajectory)
        extrapolated_input = extrapolate(last_input, second_last_input)

        new_inputs = np.hstack((result.control_input_trajectory_vector[6:], extrapolated_input))
        prev_inputs.append(new_inputs)

        last_point, second_last_point = get_last_two_elements(result.position_trajectory[:, :3])
        extrapolated_point = extrapolate(last_point, second_last_point)

        new_trajectory = np.hstack((result.position_trajectory_vector[3:], extrapolated_point))
        prev_trajectories.append(new_trajectory)
    return initial_states, prev_inputs, prev_trajectories
    
    
def main():
    settings = load_yaml_file("../../cpp/params/model_params.yaml")
    settings['MPCConfig']['delta_t'] = 1 / settings['MPCConfig']['mpc_freq']
    del settings['MPCConfig']['mpc_freq']
    
    initial_positions = {0: np.array([1,0,1]), 1: np.array([0,1,1])}
    waypoints = {0: np.array([[0.00, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00],
                            [2.25, -1.00, 0.00, 1.00, 0.00, 0.00, 0.00]]),
            1: np.array([[0.00, 0.00, 1.00, 1.00, 0.00, 0.00, 0.00],
                        [2.25, 0.00, -1.00, 1.00, 0.00, 0.00, 0.00]])}
    num_drones = len(waypoints)
    amswarm_kwargs = {
        "config": amswarm.MPCConfig(**settings['MPCConfig']),
        "weights": amswarm.MPCWeights(**settings['MPCWeights']),
        "limits": amswarm.PhysicalLimits(**settings['PhysicalLimits']),
        "dynamics": amswarm.SparseDynamics(**settings['Dynamics']),
    }
    drones = [
        amswarm.Drone(waypoints=waypoints[key],
                      initial_pos=initial_positions[key],
                      **amswarm_kwargs) for key in waypoints
    ]
    swarm = amswarm.Swarm(drones)
    num_inputs = 6 # TODO fix this
    
    position_results = []
    control_input_results = []
    for key in waypoints:
        position_results.append(initial_positions[key]) # add initial position to results
        control_input_results.append(np.empty((0,num_inputs)))
        
    initial_states, prev_inputs, prev_trajectories = initialize_swarm_optimization(
                    waypoints, settings)
    opt = [amswarm.SolveOptions()] * num_drones
    step_result = solve_swarm(swarm, 0, initial_states, prev_trajectories, opt,
                                prev_inputs)
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
        initial_states, prev_inputs, prev_trajectories = update_for_next_iteration(step_result)
        
        for i in range(len(drones)):
            position_results[i] = np.vstack((position_results[i], step_result.drone_results[i].position_trajectory[0,:]))
            control_input_results[i] = np.vstack((control_input_results[i], step_result.drone_results[i].control_input_trajectory[0,:]))
            
    formatted_string = '[' + '; '.join([str(row)[1:-1] for row in position_results[1]]) + ']'
    print(formatted_string)
    np.set_printoptions(precision=3, suppress=True, linewidth=100, threshold=np.inf)
    formatted_string = '[' + '; '.join([str(row)[1:-1] for row in control_input_results[1]]) + ']'
    print("\n")
    print(formatted_string)
    
if __name__ == "__main__":
    main()