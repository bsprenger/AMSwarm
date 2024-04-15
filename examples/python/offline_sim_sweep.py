import numpy as np
from run_offline_sim import run_offline_sim
from scipy import io

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
    np.random.seed(2)
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
    # waypoints = generate_random_waypoints(num_drones=1, num_waypoints=5)
    waypoints = {1: np.array([[0,1,1,1,0,0,0,0,0,0],
                           [4,-1,-1,1,0,0,0,0,0,0]]),
                2: np.array([[0,-1,1,1,0,0,0,0,0,0],
                           [4,1,-1,1,0,0,0,0,0,0]])}
    # print(waypoints)
    simulation_results = run_offline_sim(waypoints)
    simulation_results["waypoints"] = waypoints[1]
    io.savemat('test.mat', simulation_results)
    print("Done.")

if __name__ == "__main__":
    main()