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

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    settings = load_yaml_file("../../cpp/params/model_params.yaml")
    # settings['MPCConfig']['delta_t'] = 1 / settings['MPCConfig']['mpc_freq']
    # del settings['MPCConfig']['mpc_freq']
    
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
    constraint_configs = [amswarm.ConstraintConfig() for k in waypoints]
    
    print("Initially, the drones assume the other drones are staying in their initial positions,\n because none of the drones has planned a path yet.")
    print("So, we plan initial paths:")
    solve_status, drone_results = swarm.solve(0, initial_states, drone_results, constraint_configs)
    print(f"Solve status: {solve_status}")
    print(f"Drone 1 position:\n{drone_results[0].input_position_trajectory}")
    print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    print("Notice how these paths intersect. This is because the drones are not aware of each other's paths.")
    
    print("Now, we can re-run the initial solve with this new information of the other drone's path:")
    solve_status, drone_results = swarm.solve(0, initial_states, drone_results, constraint_configs)
    print(f"Solve status: {solve_status}")
    print(f"Drone 1 position:\n{drone_results[0].state_trajectory}")
    print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    print("Notice that drone 2 now will avoid drone 1 now.\nThis is still the initial solve, so the drones start at their initial positions.")
    
    print("Now, we can advance the drones' positions, update the estimate for their trajectories (by extrapolating from the previous solve), and re-solve at the next times step:")
    initial_states = [extract_next_state_from_result(result) for result in drone_results]
    [result.advanceForNextSolveStep() for result in drone_results]
    solve_status, drone_results = swarm.solve(0.125, initial_states, drone_results, constraint_configs)
    print(f"Solve status: {solve_status}")
    print(f"Drone 1 position:\n{drone_results[0].state_trajectory}")
    print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    
    print("and again:")
    initial_states = [extract_next_state_from_result(result) for result in drone_results]
    [result.advanceForNextSolveStep() for result in drone_results]
    solve_status, drone_results = swarm.solve(0.25, initial_states, drone_results, constraint_configs)
    print(f"Solve status: {solve_status}")
    print(f"Drone 1 position:\n{drone_results[0].state_trajectory}")
    print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    
    print("We can also disable the hard waypoint constraints and see that they miss the waypoints now:")
    [cfg.setWaypointsConstraints(False, False, False) for cfg in constraint_configs]
    initial_states = [extract_next_state_from_result(result) for result in drone_results]
    [result.advanceForNextSolveStep() for result in drone_results]
    solve_status, drone_results = swarm.solve(0.25, initial_states, drone_results, constraint_configs)
    print(f"Solve status: {solve_status}")
    print(f"Drone 1 position:\n{drone_results[0].state_trajectory}")
    print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    
    
if __name__ == "__main__":
    main()