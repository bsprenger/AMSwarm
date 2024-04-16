"""This script is an example of how to generate random waypoints and run an online simulation
with feedback from gym-pybullet-drones. 
Note that truly random waypoints are liable to make the drones perform very poorly,
as they will almost certainly not be able to track arbitrary waypoints paths.
To properly test, you should define your own waypoints that are at least somewhat
feasible for the drones to track."""
import numpy as np
from run_online_sim import run_online_sim
from utils import generate_random_waypoints
from logger import FileLogger
from pathlib import Path

def main():
    base_path = Path(__file__).resolve().parent / "data"
    base_path.mkdir(parents=True, exist_ok=True)
    log_file = base_path / "run1.json"
    logger = FileLogger(log_file)

    waypoints = generate_random_waypoints(num_drones=1, num_waypoints=5)
    
    simulation_results = run_online_sim(waypoints, gui=False)
    logger.log(simulation_results)

    print("Completed simulation.")

if __name__ == "__main__":
    main()