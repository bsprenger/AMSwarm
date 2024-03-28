import numpy as np
from run_online_sim import run_online_sim
from logger import FileLogger
from pathlib import Path

def generate_time_stamps(num_waypoints, min_interval=3.0, max_interval=4):
    """
    Generate a sorted array of time stamps starting at 0, with specified constraints.
    """
    if num_waypoints <= 1:
        return np.array([0])
    
    intervals = np.random.uniform(min_interval, max_interval, size=num_waypoints-1)
    time_stamps = np.insert(np.cumsum(intervals), 0, 0)
    return time_stamps

def generate_random_waypoints(num_drones=3, num_waypoints=2, min_distance=0.5):
    """
    Generate a dictionary of random waypoints for a given number of drones, ensuring
    matching times for all drones that are randomly distributed, start at time 0,
    and have positions within specified ranges, with drones starting some minimum distance apart.
    """
    waypoints = {}
    time_stamps = generate_time_stamps(num_waypoints)
    starting_positions = []
    
    for drone_id in range(1, num_drones + 1):
        valid_position = False
        while not valid_position:
            # Generate x, y positions within [-2, 2] range
            x, y = np.random.rand(2) * 1 - 0.5
            # Generate z position within [0, 2] range as a scalar directly
            z = np.random.rand() * 1  # This ensures z is a scalar
            position = np.array([x, y, z])
            
            # Check if position is sufficiently far from all other starting positions
            if all(np.linalg.norm(position - other_position) >= min_distance for other_position in starting_positions):
                valid_position = True
                starting_positions.append(position)
        
        drone_waypoints = np.zeros((num_waypoints, 10))  # Initialize waypoint array
        drone_waypoints[:, 0] = time_stamps  # Set time stamps
        drone_waypoints[0, 1:4] = position  # Set starting position for the first waypoint
        # For subsequent waypoints, generate positions randomly within the allowed ranges
        drone_waypoints[1:, 1:3] = np.random.rand(num_waypoints-1, 2) * 1 - 0.5
        drone_waypoints[1:, 3] = np.random.rand(num_waypoints-1) * 1
        
        waypoints[drone_id] = drone_waypoints
    
    return waypoints

def main():
    base_path = Path(__file__).resolve().parent / "data"
    num_waypoints = 10  # Set to 30 waypoints for each simulation
    
    # Define the batches for simulation runs: first with 3 drones, then 6, and so on up to 15
    drone_batches = [3]
    runs_per_batch = 1  # 20 runs for each batch

    for num_drones in drone_batches:
        # Create a subfolder for each batch
        batch_path = base_path
        batch_path.mkdir(parents=True, exist_ok=True)

        for run in range(1, runs_per_batch + 1):
            log_file = batch_path / f"{num_drones}_drones_run_{run}.json"
            logger = FileLogger(log_file)
            waypoints = generate_random_waypoints(num_drones=num_drones, num_waypoints=num_waypoints)
            # for k, v in waypoints.items():
            #     simulation_results = run_online_sim({k: v}, False)
            #     logger.log(simulation_results)
            simulation_results = run_online_sim(waypoints, True)
            logger.log(simulation_results)
            print(f"Completed: {num_drones} drones, Run {run}")

if __name__ == "__main__":
    main()