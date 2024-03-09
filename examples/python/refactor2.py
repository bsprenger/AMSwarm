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
    
    initial_positions = {1: np.array([1,1,1]), 2: np.array([-1,1,1]), 3: np.array([0,-1,1])}
    waypoints = {1: np.array([[ 0.        ,  1.        ,  1.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.90616986,  1.        ,  0.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 2.68726233,  0.8       ,  0.2       ,  1.3       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 3.65592666,  0.6       ,  0.4       ,  1.2       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 5.12454677,  0.4       ,  0.6       ,  1.4       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 6.12445833,  0.2       ,  0.8       ,  1.3       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 7.4680895 ,  0.        ,  1.        ,  1.5       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 7.8118091 , -0.2       ,  0.8       ,  1.8       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [10.43657695, -0.4       ,  0.6       ,  1.7       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [11.31149957, -0.6       ,  0.4       ,  1.4       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [11.5927247 , -0.8       ,  0.2       ,  1.5       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [12.37390561, -1.        ,  0.        ,  1.6       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [13.68628954, -0.8       , -0.2       ,  1.3       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [15.0299207 , -0.6       , -0.4       ,  1.4       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [16.15482121, -0.4       , -0.6       ,  1.7       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [17.3422162 , -0.2       , -0.8       ,  1.6       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [18.373375  ,  0.        , -1.        ,  1.3       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [18.90457801,  0.2       , -0.8       ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]), 2: np.array([[ 0.        , -1.        ,  1.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.90616986, -1.        ,  0.        ,  1.1       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 2.68726233, -0.8       , -0.2       ,  1.4       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 3.65592666, -0.6       , -0.4       ,  1.1       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 5.12454677, -0.4       , -0.6       ,  1.5       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 6.12445833, -0.2       , -0.8       ,  1.2       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 7.4680895 ,  0.        , -1.        ,  1.6       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 7.8118091 ,  0.2       , -0.8       ,  1.9       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [10.43657695,  0.4       , -0.6       ,  1.6       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [11.31149957,  0.6       , -0.4       ,  1.3       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [11.5927247 ,  0.8       , -0.2       ,  1.6       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [12.37390561,  1.        ,  0.        ,  1.5       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [13.68628954,  0.8       ,  0.2       ,  1.2       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [15.0299207 ,  0.6       ,  0.4       ,  1.5       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [16.15482121,  0.4       ,  0.6       ,  1.8       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [17.3422162 ,  0.2       ,  0.8       ,  1.5       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [18.373375  ,  0.        ,  1.        ,  1.2       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [18.90457801, -0.2       ,  0.8       ,  1.1       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]), 3: np.array([[ 0.        ,  0.        , -1.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.90616986,  0.        , -1.        ,  1.2       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 2.68726233,  0.2       , -0.8       ,  1.5       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 3.65592666,  0.4       , -0.6       ,  1.3       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 5.12454677,  0.6       , -0.4       ,  1.6       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 6.12445833,  0.8       , -0.2       ,  1.4       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 7.4680895 ,  1.        ,  0.        ,  1.7       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 7.8118091 ,  0.8       ,  0.2       ,  2.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [10.43657695,  0.6       ,  0.4       ,  1.5       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [11.31149957,  0.4       ,  0.6       ,  1.2       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [11.5927247 ,  0.2       ,  0.8       ,  1.7       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [12.37390561,  0.        ,  1.        ,  1.4       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [13.68628954, -0.2       ,  0.8       ,  1.3       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [15.0299207 , -0.4       ,  0.6       ,  1.6       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [16.15482121, -0.6       ,  0.4       ,  1.9       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [17.3422162 , -0.8       ,  0.2       ,  1.4       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [18.373375  , -1.        ,  0.        ,  1.1       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [18.90457801, -0.8       , -0.2       ,  1.2       ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])}

    drone_results = [amswarm.DroneResult.generateInitialDroneResult(initial_positions[k], settings['MPCConfig']['K']) for k in waypoints]
    print(drone_results[0].input_position_trajectory)
    amswarm_kwargs = {
        "method": amswarm.UpdateMethod.Lagrange,
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
    initial_states = [np.array([1,1,1,0,0,0]), np.array([-1,1,1,0,0,0]), np.array([0,-1,1,0,0,0])]
    constraint_configs = [amswarm.ConstraintConfig() for k in waypoints]
    [cfg.setWaypointsConstraints(True, False, False) for cfg in constraint_configs]
    solve_status, drone_results = swarm.solve(0, initial_states, drone_results, constraint_configs)
    print(solve_status)
    print(drone_results[0].input_position_trajectory)
    # print(f"Solve status: {solve_status}")
    # print(f"Drone 1 position:\n{drone_results[0].state_trajectory}")
    # print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    # print("Notice how these paths intersect. This is because the drones are not aware of each other's paths.")
    
    # print("Now, we can re-run the initial solve with this new information of the other drone's path:")
    # solve_status, drone_results = swarm.solve(0, initial_states, drone_results, constraint_configs)
    # print(f"Solve status: {solve_status}")
    # print(f"Drone 1 position:\n{drone_results[0].state_trajectory}")
    # print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    # print("Notice that drone 2 now will avoid drone 1 now.\nThis is still the initial solve, so the drones start at their initial positions.")
    
    # print("Now, we can advance the drones' positions, update the estimate for their trajectories (by extrapolating from the previous solve), and re-solve at the next times step:")
    # initial_states = [extract_next_state_from_result(result) for result in drone_results]
    # [result.advanceForNextSolveStep() for result in drone_results]
    # solve_status, drone_results = swarm.solve(0.125, initial_states, drone_results, constraint_configs)
    # print(f"Solve status: {solve_status}")
    # print(f"Drone 1 position:\n{drone_results[0].state_trajectory}")
    # print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    
    # print("and again:")
    # initial_states = [extract_next_state_from_result(result) for result in drone_results]
    # [result.advanceForNextSolveStep() for result in drone_results]
    # solve_status, drone_results = swarm.solve(0.25, initial_states, drone_results, constraint_configs)
    # print(f"Solve status: {solve_status}")
    # print(f"Drone 1 position:\n{drone_results[0].state_trajectory}")
    # print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    
    # print("We can also disable the hard waypoint constraints and see that they miss the waypoints now:")
    # [cfg.setWaypointsConstraints(False, False, False) for cfg in constraint_configs]
    # initial_states = [extract_next_state_from_result(result) for result in drone_results]
    # [result.advanceForNextSolveStep() for result in drone_results]
    # solve_status, drone_results = swarm.solve(0.25, initial_states, drone_results, constraint_configs)
    # print(f"Solve status: {solve_status}")
    # print(f"Drone 1 position:\n{drone_results[0].state_trajectory}")
    # print(f"Drone 2 position:\n{drone_results[1].state_trajectory}")
    
    
if __name__ == "__main__":
    main()