import amswarm
import numpy as np
import yaml
from pathlib import Path

def load_yaml_file(file_path: str) -> dict:
    """Load YAML configuration file."""
    with open(Path(__file__).resolve().parent / file_path, "r") as f:
        return yaml.safe_load(f)
    
def extract_next_state_from_result(result: amswarm.DroneResult) -> np.ndarray:
    """Extract the next state from the result."""
    return result.state_trajectory[1,:]
  
def solve_swarm(swarm: amswarm.Swarm, current_time: float, initial_states,
                input_drone_results, constraint_configs):
    """Solve the swarm optimization problem. If it fails, disable constraints and try again."""
    [cfg.setWaypointsConstraints(False, False, False) for cfg in constraint_configs]
    # [cfg.setInputContinuityConstraints(False) for cfg in constraint_configs]
    solve_success, iters, drone_results = swarm.solve(current_time, initial_states, input_drone_results,
                                               constraint_configs)
    # print(iters)
    failed_drones = [index for index, success in enumerate(solve_success) if not success]
    if failed_drones:
        [
            cfg.setWaypointsConstraints(False, False, False)
            for index, cfg in enumerate(constraint_configs)
            if index in failed_drones
        ]
        solve_success, iters, drone_results = swarm.solve(current_time, initial_states,
                                                   input_drone_results, constraint_configs)
    return drone_results

def run_offline_sim(waypoints):
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    settings = load_yaml_file("../../cpp/params/model_params.yaml")
    
    num_drones = len(waypoints)
    last_timestamp = waypoints[list(waypoints.keys())[0]][-1, 0]
    duration_sec = round(last_timestamp * settings["MPCConfig"]["mpc_freq"]) / settings["MPCConfig"]["mpc_freq"]
    num_steps = int(duration_sec * settings["MPCConfig"]["mpc_freq"])

    results = {}
    results["position"] = np.zeros((num_steps, 3, num_drones))
    results["control"] = np.zeros((num_steps, 6, num_drones))
    
    initial_positions = {k: waypoints[k][0, 1:4] for k in waypoints}

    drone_results = [amswarm.DroneResult.generateInitialDroneResult(initial_positions[k], settings['MPCConfig']['K']) for k in waypoints]
    amswarm_kwargs = {
        "solverConfig": amswarm.AMSolverConfig(**settings['AMSolverConfig']),
        "mpcConfig": amswarm.MPCConfig(**settings['MPCConfig']),
        "weights": amswarm.MPCWeights(**settings['MPCWeights']),
        "limits": amswarm.PhysicalLimits(**settings['PhysicalLimits']),
        "dynamics": amswarm.SparseDynamics(**settings['Dynamics']),
    }
    drones = [
        amswarm.Drone(waypoints=waypoints[key],
                      **amswarm_kwargs) for key in waypoints
    ]
    swarm = amswarm.Swarm(drones)
    initial_states = [np.concatenate((initial_positions[k], [0,0,0])) for k in waypoints]
    constraint_configs = [amswarm.ConstraintConfig() for k in waypoints]
    
    # Initial guess
    drone_results = solve_swarm(swarm, 0, initial_states, drone_results, constraint_configs)
    
    # Simulate
    last_timestamp = waypoints[list(waypoints.keys())[0]][-1,-0]
    duration_sec = round(last_timestamp * settings["MPCConfig"]["mpc_freq"]) / settings["MPCConfig"]["mpc_freq"]
    for i in range(num_steps):
        # Record initial state
        for drone_index, state in enumerate(initial_states):
            results["position"][i, :, drone_index] = state[:3]
        # Solve for optimal input at current time
        drone_results = solve_swarm(swarm, 0.125*i, initial_states, drone_results, constraint_configs)
        # Record optimal input at current time
        for drone_index, drone_result in enumerate(drone_results):
            results["control"][i, 0:3, drone_index] = drone_result.input_position_trajectory[0,:]
            results["control"][i, 3:6, drone_index] = drone_result.input_velocity_trajectory[0,:]

        initial_states = [extract_next_state_from_result(result) for result in drone_results]
        [result.advanceForNextSolveStep() for result in drone_results]
        
    return results
    
    
if __name__ == "__main__":
    run_offline_sim()