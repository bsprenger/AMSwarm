import numpy as np
from run_offline_sim import run_offline_sim 

def generate_time_stamps(num_waypoints, min_interval=0.125, max_interval=4):
    """
    Generate a sorted array of time stamps starting at 0, with specified constraints.
    """
    if num_waypoints <= 1:
        return np.array([0])
    
    intervals = np.random.uniform(min_interval, max_interval, size=num_waypoints-1)
    time_stamps = np.insert(np.cumsum(intervals), 0, 0)
    return time_stamps

def generate_random_waypoints(num_drones=3, num_waypoints=2):
    """
    Generate a dictionary of random waypoints for a given number of drones, ensuring
    matching times for all drones that are randomly distributed, start at time 0,
    and have positions within specified ranges.
    """
    waypoints = {}
    time_stamps = generate_time_stamps(num_waypoints)
    
    for drone_id in range(1, num_drones + 1):
        drone_waypoints = np.zeros((num_waypoints, 10))  # Initialize waypoint array
        drone_waypoints[:, 0] = time_stamps  # Set time stamps
        
        # Generate x, y positions within [-9, 9] range
        drone_waypoints[:, 1:3] = np.random.rand(num_waypoints, 2) * 4 - 2
        
        # Generate z position within [0, 9] range
        drone_waypoints[:, 3] = np.random.rand(num_waypoints) * 2
        
        waypoints[drone_id] = drone_waypoints
    
    return waypoints

def main():
    waypoints = generate_random_waypoints(num_drones=3, num_waypoints=2)
    simulation_results = run_offline_sim(waypoints)

if __name__ == "__main__":
    main()