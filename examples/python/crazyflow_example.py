"""
Crazyflow Simulator Example

This example demonstrates the integration of AMSwarm with the crazyflow simulator.
Crazyflow is a fast, parallelizable simulation framework for Crazyflie drones built 
with JAX and MuJoCo.

For now, this example simply imports both packages to verify the integration works.
Future enhancements could include:
- Setting up a simulation environment with crazyflow
- Using AMSwarm trajectory planning with crazyflow dynamics
- Running multi-drone swarm simulations

Requirements:
    Install with crazyflow optional dependency:
    pip install -e .[crazyflow]

Usage:
    python examples/python/crazyflow_example.py
"""

import amswarm
import crazyflow


def main():
    """Main function to demonstrate crazyflow and amswarm integration."""
    print("Successfully imported amswarm and crazyflow!")
    print(f"AMSwarm module: {amswarm}")
    print(f"Crazyflow module: {crazyflow}")
    print("\nThis example demonstrates that both packages can be imported together.")
    print("Future versions will include actual simulation examples.")


if __name__ == "__main__":
    main()
