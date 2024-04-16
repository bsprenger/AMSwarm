import numpy as np
from run_offline_sim import run_offline_sim
from generate_random_waypoints import generate_random_waypoints
from logger import FileLogger
from pathlib import Path

def main():
    base_path = Path(__file__).resolve().parent / "data"
    base_path.mkdir(parents=True, exist_ok=True)
    log_file = base_path / "run1.json"
    logger = FileLogger(log_file)

    waypoints = generate_random_waypoints(num_drones=1, num_waypoints=5)
    
    simulation_results = run_offline_sim(waypoints)
    logger.log(simulation_results)

    print("Completed simulation.")

if __name__ == "__main__":
    main()