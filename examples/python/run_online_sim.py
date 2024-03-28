import amswarm
import numpy as np
import yaml
from pathlib import Path
import time

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

def load_yaml_file(file_path: str) -> dict:
    """Load YAML configuration file."""
    with open(Path(__file__).resolve().parent / file_path, "r") as f:
        return yaml.safe_load(f)
    
def initialize_simulation_environment(
    simulation_freq: float, aggregate_phy_steps: int, waypoints: dict, gui: bool
) -> CtrlAviary:
    """Initialize the simulation environment for drone control."""
    num_drones = len(waypoints)
    simulation_params = {
        "drone_model": DroneModel("cf2x"),
        "num_drones": num_drones,
        "neighbourhood_radius": 10,
        "initial_xyzs": np.vstack([waypoints[drone_id][0, 1:4] for drone_id in waypoints]),
        "physics": Physics("pyb"),
        "freq": simulation_freq,
        "aggregate_phy_steps": aggregate_phy_steps,
        "gui": gui,
        "record": False,
        "obstacles": False,
        "user_debug_gui": False,
    }
    return CtrlAviary(**simulation_params)

def initialize_drone_controllers(num_drones: int, drone_model: DroneModel) -> list:
    """Initialize PID controllers for each drone."""
    return [DSLPIDControl(drone_model) for _ in range(num_drones)]
    
def extract_next_state_from_result(result: amswarm.DroneResult) -> np.ndarray:
    """Extract the next state from the result."""
    return result.state_trajectory[1,:]
  
def solve_swarm(swarm: amswarm.Swarm, current_time: float, initial_states,
                input_drone_results, constraint_configs):
    """Solve the swarm optimization problem. If it fails, disable constraints and try again."""
    [cfg.setWaypointsConstraints(True, False, False) for cfg in constraint_configs]
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

def run_online_sim(waypoints, gui = True):
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    settings = load_yaml_file("../../cpp/params/model_params.yaml")
    drone_ids = list(waypoints.keys())
    # Setup frequencies
    simulation_freq = settings["simulation_frequency"]  # gym sim freq
    control_freq = settings["mellinger_frequency"]
    amswarm_freq = settings["MPCConfig"]["mpc_freq"]

    # Setup sim loop parameters
    aggregate_phy_steps = int(round(simulation_freq / control_freq))
    ctrl_every_n_steps = int(np.floor(simulation_freq / control_freq))
    amswarm_every_n_steps = int(np.floor(simulation_freq / amswarm_freq))
    
    # Initialize sim environment and controllers. TODO change PID to Mellinger
    num_drones = len(waypoints)
    action = {str(i): np.zeros(4) for i in range(num_drones)}
    env = initialize_simulation_environment(simulation_freq, aggregate_phy_steps, waypoints, gui)
    ctrl = initialize_drone_controllers(num_drones, DroneModel("cf2x"))
    logging_freq = int(simulation_freq / aggregate_phy_steps)
    
    last_timestamp = waypoints[list(waypoints.keys())[0]][-1, 0]
    duration_sec = round(last_timestamp * settings["MPCConfig"]["mpc_freq"]) / settings["MPCConfig"]["mpc_freq"]
    num_steps = int(duration_sec * settings["MPCConfig"]["mpc_freq"])
    results = np.zeros((num_steps, 3, num_drones))
    
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
    obs = {
        str(i): {"state": np.hstack([v[0, 1:4], np.zeros(3), np.ones(1), np.zeros(13)]).reshape(20)}
        for i, (_, v) in enumerate(waypoints.items())
    }
    initial_states = [np.concatenate((initial_positions[k], [0,0,0])) for k in waypoints]
    constraint_configs = [amswarm.ConstraintConfig() for k in waypoints]
    
    # logging variables
    states = []
    control_inputs = []
    # Initial guess
    drone_results = solve_swarm(swarm, 0, initial_states, drone_results, constraint_configs)
    
    # Simulate
    t_start = time.time()
    for i in range(0, int(duration_sec * simulation_freq), aggregate_phy_steps):
        current_time = i / simulation_freq

        # Calculate AMSwarm pos-vel reference
        if i % amswarm_every_n_steps == 0:
            initial_states = [
                np.concatenate(
                    (obs[str(drone_index)]["state"][0:3], obs[str(drone_index)]["state"][10:13])
                )
                for drone_index in range(num_drones)
            ]
            drone_results = solve_swarm(
                swarm, current_time, initial_states, drone_results, constraint_configs
            )
            # Log inputs
            control_inputs.append(
                {
                    int(drone_ids[i]): np.concatenate(
                        (
                            result.input_position_trajectory[0, :],
                            result.input_velocity_trajectory[0, :],
                        )
                    )
                    for i, result in enumerate(drone_results)
                }
            )
            current_input_pos = [result.input_position_trajectory[0, :] for result in drone_results]
            current_input_vel = [result.input_velocity_trajectory[0, :] for result in drone_results]
            [result.advanceForNextSolveStep() for result in drone_results]

        # Calculate lower-level control (thrust commands)
        if i % ctrl_every_n_steps == 0:
            for j in range(num_drones):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=ctrl_every_n_steps * env.TIMESTEP,
                    state=obs[str(j)]["state"],
                    target_pos=current_input_pos[j],
                    target_vel=current_input_vel[j],
                )

        # Log states
        states.append({int(drone_id): obs["state"] for drone_id, obs in obs.items()})

        # Apply control input and simulate one step
        obs, _, _, _ = env.step(action)
        gui and sync(i, t_start, env.TIMESTEP)
    env.close()
        
    num_steps = int(duration_sec * simulation_freq)
    timestamps = np.array([i / simulation_freq for i in range(0, num_steps, aggregate_phy_steps)])

    sim_log = {
        "num_drones": num_drones,
        "log_freq": logging_freq,
        "sim_freq": simulation_freq,
        "timestamps": timestamps,
        "states": states,
        "controls": control_inputs,
        "waypoints": waypoints,
    }
    
    return sim_log
    
    
if __name__ == "__main__":
    run_online_sim()