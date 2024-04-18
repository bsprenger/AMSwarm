"""
Drone Swarm OFFLINE Simulation Module

Usage:
This module is not intended to be run as a standalone script. Instead, it should
be imported and used within other Python files or projects that require simulation
of drone swarm behavior. The core functionality is encapsulated in the
`run_offline_sim(waypoints)`function, which takes a dictionary of drone IDs
mapped to their respective waypoints.

Example:
    ```
    from run_offline_sim import run_offline_sim
    
    # Define waypoints for 2 drone swarm - a single waypoint is [time, x, y, z, vx, vy, vz, ax, ay, az]
    waypoints = {1: np.array([[0,1,1,1,0,0,0,0,0,0],
                           [4,-1,-1,1,0,0,0,0,0,0]]),
                2: np.array([[0,-1,1,1,0,0,0,0,0,0],
                           [4,1,-1,1,0,0,0,0,0,0]])}
    
    # Run the simulation
    simulation_results = run_offline_sim(waypoints)
    ```
"""
import amswarm
import numpy as np
from utils import load_yaml_file
    
def extract_next_state_from_result(result: amswarm.DroneResult) -> np.ndarray:
    """
    Extracts the next state of a single drone from an optimization result. The
    optimization result returns the state trajectory over the horizon starting 
    from the current time step. When we simulate offline, we are assuming that
    the drone will perfectly be able to follow the optimal trajectory. Thus,
    the next state of the drone is the second state in the trajectory.
    This function therefore extracts the second state from the state trajectory.

    Args:
        result (amswarm.DroneResult): An object containing optimization results for a drone.

    Returns:
        np.ndarray: The next state of the drone as a numpy array.
    """
    return result.state_trajectory[1,:]
  
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

def run_offline_sim(waypoints):
    """
    Runs an offline simulation of the drone swarm based on predefined waypoints.
    The offline simulation essentially assumes that drones are perfectly able to
    track predicted trajectories that are output by the AMSwarm algorithm. 
    Therefore, at each time step, the optimal input for each drone is calculated
    and the drone's state is advanced to the predicted state at the next time step.
    Note that the solver output predicts the state over the entire horizon starting
    from the current time step. The next state of the drone is therefore the second
    state in the trajectory.

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
    
    # Initialize simulation parameters based on waypoints
    num_drones = len(waypoints)
    last_timestamp = waypoints[list(waypoints.keys())[0]][-1, 0]
    duration_sec = round(last_timestamp * settings["MPCConfig"]["mpc_freq"]) / settings["MPCConfig"]["mpc_freq"]
    num_steps = int(duration_sec * settings["MPCConfig"]["mpc_freq"])
    initial_positions = {k: waypoints[k][0, 1:4] for k in waypoints}

    # Initialize results storage
    results = {}
    results["position"] = np.zeros((num_steps, 3, num_drones))
    results["control"] = np.zeros((num_steps, 6, num_drones))

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

    # Set initial states (current position, zero velocities)
    initial_states = [np.concatenate((initial_positions[k], [0,0,0])) for k in waypoints]

    # Create constraint config objects. NOTE these are modified in solve_swarm
    constraint_configs = [amswarm.ConstraintConfig() for k in waypoints]
    
    # The drones' initial states are set to their current positions and zero velocities.
    # Therefore, the first time we try to solve the optimization problem, they won't
    # consider collisions, because each drone will think the others are stationary
    # at their current positions. To prevent this, we run a single initial solve step,
    # where the drones will not consider collisions. Then, from that point on, they
    # have reasonable guesses over where each other drone's trajectory, and they
    # will be able to consider collisions by avoiding the predicted trajectories.
    drone_results = solve_swarm(swarm, 0, initial_states, drone_results, constraint_configs)
    
    ## --- MAIN SIMULATION LOOP --- ##
    for i in range(num_steps):
        # Record current state
        for drone_index, state in enumerate(initial_states):
            results["position"][i, :, drone_index] = state[:3]
        # Solve for optimal input at current time
        drone_results = solve_swarm(swarm, 0.125*i, initial_states, drone_results, constraint_configs)
        # Record optimal input at current time
        for drone_index, drone_result in enumerate(drone_results):
            results["control"][i, 0:3, drone_index] = drone_result.input_position_trajectory[0,:]
            results["control"][i, 3:6, drone_index] = drone_result.input_velocity_trajectory[0,:]
        # Advance initial state to next predicted state (essentially assuming perfect
        # tracking of predicted path)
        initial_states = [extract_next_state_from_result(result) for result in drone_results]
        # The drones need to know each other drones' predicted trajectories from the
        # CURRENT time step. The output of the solver is ALSO the predicted trajectory
        # from the CURRENT time step. Therefore, the next time through the loop,
        # we need to advance the drones' results to the next time step, so that they
        # can consider each other's predicted trajectories from the NEXT time step.
        [result.advanceForNextSolveStep() for result in drone_results]
        
    return results