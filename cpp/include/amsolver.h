#ifndef AMSOLVER_H
#define AMSOLVER_H

#include <constraint.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <vector>
#include <memory>
#include <algorithm>
#include <execution>

using namespace Eigen;

// Abstract class for solver
template<typename ResultType, typename SolverArgsType>
class AMSolver {
protected:
    std::vector<std::unique_ptr<Constraint>> constConstraints;
    std::vector<std::unique_ptr<Constraint>> nonConstConstraints;

    // cost of the form 0.5 * x^T * quadCost * x + x^T * linearCost
    SparseMatrix<double> quadCost;
    VectorXd linearCost;

    virtual void preSolve(const SolverArgsType& args) = 0;
    virtual ResultType postSolve(const VectorXd& x, const SolverArgsType& args) = 0;
    std::pair<bool, VectorXd> actualSolve(const SolverArgsType& args);

public:
    AMSolver() = default;
    virtual ~AMSolver() = default;

    void addConstraint(std::unique_ptr<Constraint> constraint, bool isConstant);
    void updateConstraints(double rho, const VectorXd& x);
    void resetConstraints();
    std::pair<bool, ResultType> solve(const SolverArgsType& args);
};


// -------------------------- IMPLEMENTATION -------------------------- //

template<typename ResultType, typename SolverArgsType>
std::pair<bool, VectorXd> AMSolver<ResultType, SolverArgsType>::actualSolve(const SolverArgsType& args) {
    resetConstraints();

    SimplicialLDLT<SparseMatrix<double>> linearSolver;

    int iters  = 0;
    double rho_init = 1.3;
    double rho = rho_init;
    int max_iters = 1000;
    bool solver_initialized = false;

    SparseMatrix<double> Q;
    VectorXd q;
    VectorXd x;

    while (iters < max_iters) {
        // Reset Q and q to the base cost
        Q = quadCost;
        q = linearCost;

        // Construct the quadratic and linear cost matrices
        for (auto& constraint : constConstraints) {
            Q += constraint->getQuadCost(rho);
            q += constraint->getLinearCost(rho);
        }
        for (auto& constraint : nonConstConstraints) {
            Q += constraint->getQuadCost(rho);
            q += constraint->getLinearCost(rho);
        }

        // Solve the linear system
        if (!solver_initialized) {
            linearSolver.analyzePattern(Q);
            solver_initialized = true;
        }
        linearSolver.factorize(Q);
        x = linearSolver.solve(-q);
        
        // Update the constraints
        updateConstraints(rho, x);

        // Check constraints satisfaction
        bool all_constraints_satisfied = std::all_of(constConstraints.begin(), constConstraints.end(),
                                                     [&x](const std::unique_ptr<Constraint>& constraint) {
                                                         return constraint->isSatisfied(x);
                                                     }) &&
                                         std::all_of(nonConstConstraints.begin(), nonConstConstraints.end(),
                                                     [&x](const std::unique_ptr<Constraint>& constraint) {
                                                         return constraint->isSatisfied(x);
                                                     });

        if (all_constraints_satisfied) {
            return {true, x}; // Exit the loop, indicate success with the bool
        }

        // Update the penalty parameter and iters
        rho *= rho_init;
        rho = std::min(rho, 5.0e5);
        iters++;
    }

    return {false, x}; // Indicate failure with the bool. Still return the vector anyways
}

template<typename ResultType, typename SolverArgsType>
void AMSolver<ResultType, SolverArgsType>::addConstraint(std::unique_ptr<Constraint> constraint, bool isConstant) {
    if (isConstant) {
        constConstraints.push_back(std::move(constraint));
    } else {
        nonConstConstraints.push_back(std::move(constraint));
    }
}

template<typename ResultType, typename SolverArgsType>
void AMSolver<ResultType, SolverArgsType>::updateConstraints(double rho, const VectorXd& x) {
    // Parallel update for constant constraints
    std::for_each(constConstraints.begin(), constConstraints.end(),
                  [rho, &x](const std::unique_ptr<Constraint>& constraint) {
                      constraint->update(rho, x);
                  });

    // Parallel update for non-constant constraints
    std::for_each(nonConstConstraints.begin(), nonConstConstraints.end(),
                  [rho, &x](const std::unique_ptr<Constraint>& constraint) {
                      constraint->update(rho, x);
                  });
}

template<typename ResultType, typename SolverArgsType>
void AMSolver<ResultType, SolverArgsType>::resetConstraints() {
    std::for_each(constConstraints.begin(), constConstraints.end(),
                  [](const std::unique_ptr<Constraint>& constraint) {
                      constraint->reset();
                  });

    std::for_each(nonConstConstraints.begin(), nonConstConstraints.end(),
                  [](const std::unique_ptr<Constraint>& constraint) {
                      constraint->reset();
                  });
}

template<typename ResultType, typename SolverArgsType>
std::pair<bool, ResultType> AMSolver<ResultType, SolverArgsType>::solve(const SolverArgsType& args) {
    nonConstConstraints.clear();
    preSolve(args); // builds new non-const. constraints
    auto [success, result] = actualSolve(args); // TODO make explicit
    return {success, postSolve(result, args)};
}

#endif // AMSOLVER_H