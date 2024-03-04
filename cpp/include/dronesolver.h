#include <amsolver.h>

struct DroneResult {
    VectorXd x;
};

struct DroneArgs {};

struct MPCConfig {
    int K = 25;
    int n = 10;
    double delta_t = 1.0/8.0;
    double bf_gamma = 1.0;

    MPCConfig() {}

    MPCConfig(int K, int n, double delta_t, double bf_gamma) : K(K), n(n), delta_t(delta_t), bf_gamma(bf_gamma) {}
};

class DroneSolver : public AMSolver<DroneResult, DroneArgs> {
public:
    DroneSolver(const MPCConfig& config);

protected:
    void preSolve(const DroneArgs& args) override;
    DroneResult postSolve(const VectorXd& x, const DroneArgs& args) override;
    std::tuple<SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>,SparseMatrix<double>> initBernsteinMatrices(const MPCConfig& config);

private:
    SparseMatrix<double> W, W_dot, W_ddot, W_input;
};