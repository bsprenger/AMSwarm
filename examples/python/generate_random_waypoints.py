import numpy as np

def generate_time_stamps(num_waypoints, min_interval=1.0, max_interval=4):
    """
    Generate a sorted array of time stamps for waypoint creation.

    Parameters:
    - num_waypoints (int): The total number of waypoints to generate time stamps for.
    - min_interval (float, optional): The minimum interval time between consecutive waypoints in seconds. Default is 1.0.
    - max_interval (float, optional): The maximum interval time between consecutive waypoints in seconds. Default is 4.0.

    Returns:
    - numpy.ndarray: An array of time stamps, starting from 0, sorted in ascending order.

    Note: If num_waypoints is 1 or less, the function returns an array containing only the initial time stamp 0.
    """
    if num_waypoints <= 1:
        return np.array([0])
    
    intervals = np.random.uniform(min_interval, max_interval, size=num_waypoints-1)
    time_stamps = np.insert(np.cumsum(intervals), 0, 0)
    return time_stamps

def generate_random_waypoints(num_drones=3, num_waypoints=2, min_start_distance=0.5):
    """
    Generate a dictionary of random waypoints for multiple drones.

    Parameters:
    - num_drones (int, optional): The number of drones to generate waypoints for. Default is 3.
    - num_waypoints (int, optional): The number of waypoints to generate for each drone. Default is 2.
    - min_start_distance (float, optional): The minimum distance that must be maintained between the starting positions of any two drones. Default is 0.5.

    Returns:
    - dict: A dictionary where keys are drone IDs (starting from 1) and values are numpy arrays of shape (num_waypoints, 10) representing the waypoints for each drone.

    Note:
    - Waypoints are represented as numpy arrays where the first column contains time stamps for each waypoint, columns 1 to 3 contain the x, y, and z coordinates of the waypoint respectively, and the remaining columns are zeros.
    - The starting positions of the drones are randomly generated such that they maintain at least the specified minimum distance from each other.
    - Subsequent waypoints for each drone after the first are generated with positions randomly distributed within specific ranges, ensuring varied flight paths.
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
            if all(np.linalg.norm(position - other_position) >= min_start_distance for other_position in starting_positions):
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