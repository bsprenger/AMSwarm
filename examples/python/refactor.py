import amswarm
import numpy as np
import yaml
import os
from pathlib import Path

def load_yaml_file(file_path: str) -> dict:
    """Load YAML configuration file."""
    with open(Path(__file__).resolve().parent / file_path, "r") as f:
        return yaml.safe_load(f)

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    settings = load_yaml_file("../../cpp/params/model_params.yaml")
    settings['MPCConfig']['delta_t'] = 1 / settings['MPCConfig']['mpc_freq']
    del settings['MPCConfig']['mpc_freq']
    
    initial_positions = {0: np.array([0,1,1]), 1: np.array([1,0,1])}
    waypoints = {0: np.array([[0.00, 0.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                              [3.00, 0.00, -1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]),
                 1: np.array([[0.00, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                              [3.00, -1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])}

    drone_results = [amswarm.DroneResult.generateInitialDroneResult(initial_positions[k], settings['MPCConfig']['K']) for k in waypoints]
    amswarm_kwargs = {
        "config": amswarm.MPCConfig(**settings['MPCConfig']),
        "weights": amswarm.MPCWeights(**settings['MPCWeights']),
        "limits": amswarm.PhysicalLimits(**settings['PhysicalLimits']),
        "dynamics": amswarm.SparseDynamics(**settings['Dynamics']),
    }
    drones = [
        amswarm.Drone(waypoints=waypoints[key],
                      **amswarm_kwargs) for key in waypoints
    ]
    swarm = amswarm.Swarm(drones)
    initial_states = [np.array([0,1,1,0,0,0]), np.array([1,0,1,0,0,0])]
    
    solve_status, drone_results = swarm.solve(0, initial_states, drone_results, True)
    print("Initial guess means drones have no idea where the other drones are going")
    print(drone_results[0].input_position_trajectory)
    print(drone_results[1].input_position_trajectory)
    
    solve_status, drone_results = swarm.solve(0, initial_states, drone_results, True)
    print("Now the drones have a better idea of where the other drones are going")
    print(drone_results[0].input_position_trajectory)
    print(drone_results[1].input_position_trajectory)
    
    solve_status, drone_results = swarm.solve(0, initial_states, drone_results, True)
    print(solve_status)
    print("But now it won't consider a collision, as the previous trajectories are not colliding...")
    print(drone_results[0].input_position_trajectory)
    print(drone_results[1].input_position_trajectory)
    
    solve_status, drone_results = swarm.solve(0, initial_states, drone_results, True)
    print(solve_status)
    print("But now it will again.")
    print(drone_results[0].input_position_trajectory)
    print(drone_results[1].input_position_trajectory)
    
    
if __name__ == "__main__":
    main()