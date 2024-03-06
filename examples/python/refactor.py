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
    settings = load_yaml_file("../../cpp/params/model_params.yaml")
    settings['MPCConfig']['delta_t'] = 1 / settings['MPCConfig']['mpc_freq']
    del settings['MPCConfig']['mpc_freq']
    
    initial_positions = {0: np.array([0,1,1]), 1: np.array([1,0,1])}
    waypoints = {0: np.array([[0.00, 0.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                              [3.00, 0.00, -1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]),
                 1: np.array([[0.00, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                              [3.00, -1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])}
    
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
    initial_states = [np.array([0,1,1,0,0,0]), np.array([1,0,1,0,0,0])]
    prev_trajectories = [np.tile(initial_positions[0], settings["MPCConfig"]["K"]+1), np.tile(initial_positions[1], settings["MPCConfig"]["K"]+1)]
    prev_inputs = [np.zeros(6 * settings['MPCConfig']['K']), np.zeros(6 * settings['MPCConfig']['K'])] # TODO
    a,b = swarm.solve(0, initial_states, prev_trajectories, prev_inputs)    
    prev_trajectories = [b[0].position_trajectory_vector, b[1].position_trajectory_vector]
    a,b = swarm.solve(0.0, initial_states, prev_trajectories, prev_inputs)
    print(a)
    np.set_printoptions(precision=2, suppress=True, linewidth=100)
    print(b[0].state_trajectory)
    print(b[1].state_trajectory)
    
    
if __name__ == "__main__":
    main()