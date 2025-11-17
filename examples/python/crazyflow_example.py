"""
Crazyflow Simulator Example

This example demonstrates the integration of AMSwarm with the crazyflow simulator.
Crazyflow is a fast, parallelizable simulation framework for Crazyflie drones built 
with JAX and MuJoCo.

For now, this example simply imports crazyflow to verify the integration works.
Future enhancements could include:
- Setting up a simulation environment with crazyflow
- Using AMSwarm trajectory planning with crazyflow dynamics
- Running multi-drone swarm simulations

Requirements:
    Install with examples optional dependency:
    pip install -e .[examples]

Usage:
    python examples/python/crazyflow_example.py
"""

import amswarm
import crazyflow


def main():
    """Main function to demonstrate crazyflow integration."""
    print("Successfully imported amswarm and crazyflow!")
    print(f"amswarm module: {amswarm}")
    print(f"Crazyflow module: {crazyflow}")
    print(f"Crazyflow version: {crazyflow.__version__}")
    print("\nThis example demonstrates that crazyflow can be used with AMSwarm.")
    print("Future versions will include actual simulation examples with AMSwarm trajectory planning.")


if __name__ == "__main__":
    main()
