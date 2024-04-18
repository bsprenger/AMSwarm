"""
Drone Swarm ONLINE Simulation Module

This module simulates drone swarm behavior with online feedback from the
gym-pybullet-drones simulator.

Usage:
This module is not intended to be run as a standalone script. Instead, it should
be imported and used within other Python files or projects that require simulation
of drone swarm behavior. The core functionality is provided by the 
`run_online_sim(waypoints, gui)` function, which takes a dictionary of waypoints 
for each drone in the swarm and an optional graphical user interface (GUI) flag.

Example:
    ```
    from run_online_sim import run_online_sim

    # Define waypoints for 2 drone swarm - a single waypoint is [time, x, y, z, vx, vy, vz, ax, ay, az]
    waypoints = {1: np.array([[0,1,1,1,0,0,0,0,0,0],
                           [4,-1,-1,1,0,0,0,0,0,0]]),
                2: np.array([[0,-1,1,1,0,0,0,0,0,0],
                           [4,1,-1,1,0,0,0,0,0,0]])}

    # Execute the simulation with GUI enabled
    simulation_results = run_online_sim(waypoints, gui=True)
    ```
"""
import numpy as np
import yaml
from pathlib import Path
import time

import amswarm
from utils import load_yaml_file

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
    
def initialize_simulation_environment(
    simulation_freq: float, aggregate_phy_steps: int, waypoints: dict, gui: bool
) -> CtrlAviary:
    """
    Initializes the simulation environment from gym-pybullet-drones with the
    specified parameters and drone waypoints.

    Args:
        simulation_freq (float): The simulation frequency in Hz.
        aggregate_phy_steps (int): The number of physics simulation steps to aggregate for each control step.
        waypoints (dict): A dictionary mapping drone IDs to their waypoints. Each waypoint is an array with rows
                          in the format [time, x, y, z, vx, vy, vz, ax, ay, az].
        gui (bool): Whether to enable the graphical user interface.

    Returns:
        CtrlAviary: An instance of the CtrlAviary class configured with the provided parameters and initial positions
                    of drones derived from waypoints.
    """
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
    """
    Initializes Mellinger controllers for each drone in the swarm. These controllers
    are used as lower-level trajectory-tracking controllers. See thesis document
    for more details.

    Args:
        num_drones (int): The number of drones in the swarm.
        drone_model (DroneModel): The model of the drones to be used in the simulation.

    Returns:
        list: A list of DSLPIDControl instances, one for each drone.
    """
    return [DSLPIDControl(drone_model) for _ in range(num_drones)]
  
def solve_swarm(swarm: amswarm.Swarm, current_time: float, initial_states,
                input_drone_results, constraint_configs):
    """
    Solves the optimization problem for a drone swarm with backup logic in case
    the optimization fails for some drones.
    In the case of failures, the function will attempt to solve the optimization
    problem again with less constraints enabled. In this case, we disable the 
    waypoint constraints for the drones that failed the first time (leaving the
    other drones' constraints unchanged). This is done to make the optimization
    problem easier to solve for the drones that failed the first time.
    Note that for this example, only the waypoint POSITION constraint is turned
    on. The input continuity constraint is turned on by default as well.

    Args:
        swarm (amswarm.Swarm): The drone swarm object.
        current_time (float): Current time step in the simulation.
        initial_states: Initial states of the drones.
        input_drone_results: Input results from previous optimizations.
        constraint_configs: Constraint configurations for the drones.

    Returns:
        The results of the optimization attempt for each drone.
    """
    ## INITIAL SOLVE ATTEMPT
    # Enable waypoint position constraint, disable waypoint velocity and acceleration constraints
    [cfg.setWaypointsConstraints(True, False, False) for cfg in constraint_configs]
    # Disable input continuity constraints
    # [cfg.setInputContinuityConstraints(False) for cfg in constraint_configs]
    # Try to solve the optimization problem
    solve_success, iters, drone_results = swarm.solve(current_time, initial_states, input_drone_results,
                                               constraint_configs)
    # Check if any drones failed to solve
    failed_drones = [index for index, success in enumerate(solve_success) if not success]
    ## BACKUP SOLVE ATTEMPT (if necessary)
    if failed_drones:
        [
            cfg.setWaypointsConstraints(False, False, False) # Set all waypoint constraints to False - could also turn of continuity constraints if desired
            for index, cfg in enumerate(constraint_configs)
            if index in failed_drones
        ]
        solve_success, iters, drone_results = swarm.solve(current_time, initial_states,
                                                   input_drone_results, constraint_configs)
    return drone_results

def run_online_sim(waypoints, gui = True):
    """
    Runs an online simulation of the drone swarm based on predefined waypoints.
    The simulation gets state feedback at each time step from gym-pybullet-drones. 

    Args:
        waypoints: A dictionary mapping drone identifiers to their waypoints.

    Returns:
        A dictionary containing the positions and control inputs for each drone
        at each timestep of the simulation.
    """
    # Set numpy print options for better readability
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    # Load simulation settings from a YAML file
    settings = load_yaml_file("../../cpp/params/model_params.yaml")

    # Extract drone info from waypoints
    drone_ids = list(waypoints.keys())
    num_drones = len(waypoints)
    initial_positions = {k: waypoints[k][0, 1:4] for k in waypoints}

    # Setup frequencies
    simulation_freq = settings["simulation_frequency"]  # gym sim freq
    control_freq = settings["mellinger_frequency"]
    amswarm_freq = settings["MPCConfig"]["mpc_freq"]

    # Setup sim loop parameters
    aggregate_phy_steps = int(round(simulation_freq / control_freq))
    ctrl_every_n_steps = int(np.floor(simulation_freq / control_freq))
    amswarm_every_n_steps = int(np.floor(simulation_freq / amswarm_freq))
    logging_freq = int(simulation_freq / aggregate_phy_steps)
    
    # Initialize sim environment and controllers
    action = {str(i): np.zeros(4) for i in range(num_drones)}
    env = initialize_simulation_environment(simulation_freq, aggregate_phy_steps, waypoints, gui)
    ctrl = initialize_drone_controllers(num_drones, DroneModel("cf2x"))
    
    # Determine simulation duration and steps
    last_timestamp = waypoints[list(waypoints.keys())[0]][-1, 0]
    duration_sec = round(last_timestamp * settings["MPCConfig"]["mpc_freq"]) / settings["MPCConfig"]["mpc_freq"]
    num_steps = int(duration_sec * settings["MPCConfig"]["mpc_freq"])
    
    # Initialize dummy drone results. At each solve step, the solver must take 
    # the previous drone results as input, because the continuity constraints
    # need to enforce continuity with previous inputs. Since at the initial solve
    # step, there are no previous inputs to enforce continuity with, we initialize
    # the previous input trajectory to be the drones' current positions. This way,
    # the drones' first actual input will be continuous with their current position,
    # preventing sharp initial inputs which cause them to crash straight away.
    drone_results = [amswarm.DroneResult.generateInitialDroneResult(initial_positions[k], settings['MPCConfig']['K']) for k in waypoints]

    # Prepare keyword arguments for Drone and Swarm initialization
    amswarm_kwargs = {
        "solverConfig": amswarm.AMSolverConfig(**settings['AMSolverConfig']),
        "mpcConfig": amswarm.MPCConfig(**settings['MPCConfig']),
        "weights": amswarm.MPCWeights(**settings['MPCWeights']),
        "limits": amswarm.PhysicalLimits(**settings['PhysicalLimits']),
        "dynamics": amswarm.SparseDynamics(**settings['Dynamics']),
    }
    # Initialize drones and swarm
    drones = [
        amswarm.Drone(waypoints=waypoints[key],
                      **amswarm_kwargs) for key in waypoints
    ]
    swarm = amswarm.Swarm(drones)

    # Get initial observations (current state measurements) and convert them to the format
    # expected by the AMSwarm solver. The state vector is [x, y, z, vx, vy, vz]
    # During the simulation these are given by the simulator. For now we set them
    # to the (stationary) initial positions of the drones
    obs = {
        str(i): {"state": np.hstack([v[0, 1:4], np.zeros(3), np.ones(1), np.zeros(13)]).reshape(20)}
        for i, (_, v) in enumerate(waypoints.items())
    }
    initial_states = [np.concatenate((initial_positions[k], [0,0,0])) for k in waypoints]

    # Create constraint config objects. NOTE These are modified within solve_swarm
    constraint_configs = [amswarm.ConstraintConfig() for k in waypoints]
    
    # logging variables
    states = [] # TODO make this consistent between offline and online
    control_inputs = []

    # The drones' initial states are set to their current positions and zero velocities.
    # Therefore, the first time we try to solve the optimization problem, they won't
    # consider collisions, because each drone will think the others are stationary
    # at their current positions. To prevent this, we run a single initial solve step,
    # where the drones will not consider collisions. Then, from that point on, they
    # have reasonable guesses over where each other drone's trajectory, and they
    # will be able to consider collisions by avoiding the predicted trajectories.
    drone_results = solve_swarm(swarm, 0, initial_states, drone_results, constraint_configs)
    
    ## --- MAIN SIMULATION LOOP --- ##
    t_start = time.time()
    for i in range(0, int(duration_sec * simulation_freq), aggregate_phy_steps):
        current_time = i / simulation_freq

        # Calculate AMSwarm pos-vel reference (optimal input)
        if i % amswarm_every_n_steps == 0:
            # Extract the drones' current states from the simulator observations
            initial_states = [
                np.concatenate(
                    (obs[str(drone_index)]["state"][0:3], obs[str(drone_index)]["state"][10:13])
                )
                for drone_index in range(num_drones)
            ]
            # Solve for optimal input at current time
            drone_results = solve_swarm(
                swarm, current_time, initial_states, drone_results, constraint_configs
            )
            # Log optimal inputs
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
            # Separate position and velocity references for input to lower-level controller
            current_input_pos = [result.input_position_trajectory[0, :] for result in drone_results]
            current_input_vel = [result.input_velocity_trajectory[0, :] for result in drone_results]
            # The drones need to know each other drones' predicted trajectories from the
            # CURRENT time step. The output of the solver is ALSO the predicted trajectory
            # from the CURRENT time step. Therefore, the next time through the loop,
            # we need to advance the drones' results to the next time step, so that they
            # can consider each other's predicted trajectories from the NEXT time step.
            [result.advanceForNextSolveStep() for result in drone_results]

        # Calculate lower-level control (thrust commands) from position and velocity references
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

        # Apply lower-level control input and simulate one step
        obs, _, _, _ = env.step(action)
        gui and sync(i, t_start, env.TIMESTEP)
    env.close()
    
    # Log simulation results
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