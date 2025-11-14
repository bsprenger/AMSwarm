#ifndef SWARM_H
#define SWARM_H

#include "drone.h"

/**
 * @file swarm.h
 * 
 * @brief Defines the `Swarm` class, which provides a convenient interface to
 * manage trajectory planning for many drones by calling Swarm::solve() rather
 * than calling Drone::solve() for each drone individually and keeping track of
 * all the predicted trajectories, passing the necessary info for collision
 * avoidance to each drone, etc.
 * 
 * @see Drone
 */

using namespace Eigen;


class Swarm {
public:
    /**
     * @brief Constructs a new Swarm instance.
     * 
     * Initializes the swarm with the given drones and prepares internal structures
     * for navigation and collision avoidance calculations.
     * 
     * @param drones A vector of shared pointers to `Drone` objects that are part of the swarm. TODO: change to unique_ptr
     */
    Swarm(std::vector<std::shared_ptr<Drone>> drones);

    /**
     * @brief Solves the navigation and collision avoidance problem for the entire swarm.
     * 
     * This method takes the current time, initial states of each drone, results from previous
     * computations, and constraint configurations to compute the next trajectories for each 
     * drone. The past results are used for drones to predict where other drones will be, so they
     * can avoid them. Also, the past results are used for the input continuity constraint.
     * 
     * @param current_time The current time.
     * @param initial_states A vector of initial states for each drone. Each initial state consists of [x, y, z, vx, vy, vz].
     * @param previous_results A vector of results from the previous computation for each drone. If there are no previous results to use, drone results can be initialized with DroneResult::generateInitialDroneResult(...)
     * @param constraint_configs A vector of constraint configurations for each drone.
     * @return A tuple containing a vector of success flags for each drone, a vector of iteration counts,
     *         and a vector of results for the current computation.
     */
    std::tuple<std::vector<bool>, std::vector<int>, std::vector<DroneResult>> solve(double current_time,
                        const std::vector<VectorXd>& initial_states,
                        const std::vector<DroneResult>& previous_results,
                        const std::vector<ConstraintConfig>& constraint_configs);

private:
    int num_drones;
    std::vector<std::shared_ptr<Drone>> drones;
    std::vector<SparseMatrix<double>> all_obstacle_envelopes; // the collision envelope for each drone at each time step

    /**
     * @brief Checks if two trajectories intersect given their positions and a matrix theta which defines the collision envelope.
     * 
     * @param traj1 The position trajectory of the first drone.
     * @param traj2 The position trajectory of the second drone.
     * @param theta The collision envelope matrix
     * @return true if the trajectories intersect, false otherwise.
     */
    bool checkIntersection(const VectorXd& traj1, const VectorXd& traj2, const SparseMatrix<double>& theta_tmp);
};

#endif