import subprocess
import time
import os

# Number of times to run the script
num_runs = 20
script_name = os.path.join(os.path.dirname(__file__), "profiling.py")

start_time = time.time()

for _ in range(num_runs):
    print(f"Run {_ + 1}/{num_runs}")
    subprocess.run(["python3", script_name])

end_time = time.time()
total_time = end_time - start_time
print(f"Total time for {num_runs} runs: {total_time} seconds")
