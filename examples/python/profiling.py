import amswarm
import numpy as np
import yaml
import os
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
    [cfg.setWaypointsConstraints(True, False, False) for cfg in constraint_configs]
    solve_success, iters, drone_results = swarm.solve(current_time, initial_states, input_drone_results,
                                               constraint_configs)
    print(iters)
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

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    settings = load_yaml_file("../../cpp/params/model_params.yaml")
    
    initial_positions = {1: np.array([1,1,1]), 2: np.array([-1,1,1]), 3: np.array([0,-1,1])}
    waypoints = {1: np.array([[ 0.        ,  1.        ,  1.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                              [4. , -1., -1., 1, 0., 0., 0., 0., 0., 0. ]]), 2: np.array([[ 0.        , -1.        ,  1.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                              [4., 1., -1., 1., 0., 0., 0., 0., 0., 0.]]), 3: np.array([[ 0.        ,  0.        , -1.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ], [4., 0., 1., 1., 0., 0., 0., 0., 0., 0.]])}

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
    initial_states = [np.array([1,1,1,0,0,0]), np.array([-1,1,1,0,0,0]), np.array([0,-1,1,0,0,0])]
    constraint_configs = [amswarm.ConstraintConfig() for k in waypoints]
    
    # Initial guess
    drone_results = solve_swarm(swarm, 0, initial_states, drone_results, constraint_configs)
    
    # Simulate
    for i in range(1, 16):
        drone_results = solve_swarm(swarm, 0.125*i, initial_states, drone_results, constraint_configs)
        initial_states = [extract_next_state_from_result(result) for result in drone_results]
        print(initial_states)
    
    
if __name__ == "__main__":
    main()