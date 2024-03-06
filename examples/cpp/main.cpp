#include <dronesolver.h
#include <iostream>
#include <Eigen/Core>

using namespace Eigen;



int main() {
    struct DroneArgs args;

    struct MPCConfig config;
    DroneSolver solver(config);
    
    // unpack results
    std::pair<bool, DroneResult> res = solver.solve(args);
    bool success = res.first;
    VectorXd result = res.second.x;

    std::cout << "Success: " << success << std::endl;
    std::cout << "Result: " << result << std::endl;

    return 0;
}