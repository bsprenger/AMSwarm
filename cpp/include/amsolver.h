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

struct AMSolverConfig {
    double rho_init;
    double max_rho;
    int max_iters;

    AMSolverConfig () {};
    AMSolverConfig(double rho_init, double max_rho, int max_iters)
        : rho_init(rho_init), max_rho(max_rho), max_iters(max_iters) {};
};

// Abstract class for solver
template<typename ResultType, typename SolverArgsType>
class AMSolver {
protected:
    std::vector<std::unique_ptr<Constraint>> constConstraints;
    std::vector<std::unique_ptr<Constraint>> nonConstConstraints;
    AMSolverConfig solverConfig;

    // cost of the form 0.5 * x^T * quadCost * x + x^T * linearCost
    SparseMatrix<double> initialQuadCost;
    VectorXd initialLinearCost;
    SparseMatrix<double> quadCost;
    VectorXd linearCost;

    virtual void preSolve(const SolverArgsType& args) = 0;
    virtual ResultType postSolve(const VectorXd& x, const SolverArgsType& args) = 0;
    std::tuple<bool, int, VectorXd> actualSolve(const SolverArgsType& args);
    void resetCostMatrices();

public:
    AMSolver(AMSolverConfig config) : solverConfig(config) {};
    virtual ~AMSolver() = default;

    void addConstraint(std::unique_ptr<Constraint> constraint, bool isConstant);
    void updateConstraints(const VectorXd& x);
    void resetConstraints();
    std::tuple<bool, int, ResultType> solve(const SolverArgsType& args);
};


// -------------------------- IMPLEMENTATION -------------------------- //
template<typename ResultType, typename SolverArgsType>
void AMSolver<ResultType, SolverArgsType>::resetCostMatrices() {
    quadCost = initialQuadCost;
    linearCost = initialLinearCost;
}

template<typename ResultType, typename SolverArgsType>
std::tuple<bool, int, VectorXd> AMSolver<ResultType, SolverArgsType>::actualSolve(const SolverArgsType& args) {
    SimplicialLDLT<SparseMatrix<double>> linearSolver;

    int iters  = 0;
    double rho = solverConfig.rho_init;
    bool solver_initialized = false;

    SparseMatrix<double> Q(quadCost.rows(), quadCost.cols());
    VectorXd q = VectorXd::Zero(quadCost.rows());
    VectorXd x = VectorXd::Zero(quadCost.rows());
    VectorXd bregmanMult = VectorXd::Zero(quadCost.rows());

    SparseMatrix<double> quadConstraintTerms(quadCost.rows(), quadCost.cols());
    VectorXd linearConstraintTerms = VectorXd::Zero(linearCost.rows());
    for (auto& constraint : constConstraints) {
        quadConstraintTerms += constraint->getQuadraticTerm();
        linearConstraintTerms += constraint->getLinearTerm();
    }
    for (auto& constraint : nonConstConstraints) {
        quadConstraintTerms += constraint->getQuadraticTerm();
        linearConstraintTerms += constraint->getLinearTerm();
    }

    while (iters < solverConfig.max_iters) {
        Q = quadCost + rho * quadConstraintTerms;
        
        // Construct the linear cost matrices
        linearConstraintTerms -= bregmanMult;
        q = linearCost + rho * linearConstraintTerms;
        
        // Solve the QP
        linearSolver.compute(Q);
        x = linearSolver.solve(-q);

        // Update the constraints
        updateConstraints(x);

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
            return std::make_tuple(true, iters, x); // Exit the loop, indicate success with the bool
        }

        // Update the penalty parameter and iters
        linearConstraintTerms.setZero();
        for (auto& constraint : constConstraints) {
            linearConstraintTerms += constraint->getLinearTerm();
        }
        for (auto& constraint : nonConstConstraints) {
            linearConstraintTerms += constraint->getLinearTerm();
        }
        VectorXd bregmanUpdate = 0.5 * (quadConstraintTerms * x + linearConstraintTerms);
        bregmanMult -= bregmanUpdate; // TODO check the direction
        
        rho *= solverConfig.rho_init;
        rho = std::min(rho, solverConfig.max_rho);
        iters++;
    }

    return std::make_tuple(false, iters, x); // Indicate failure with the bool. Still return the vector anyways
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
void AMSolver<ResultType, SolverArgsType>::updateConstraints(const VectorXd& x) {
    std::for_each(constConstraints.begin(), constConstraints.end(),
                  [&x](const std::unique_ptr<Constraint>& constraint) {
                      constraint->update(x);
                  });

    std::for_each(nonConstConstraints.begin(), nonConstConstraints.end(),
                  [&x](const std::unique_ptr<Constraint>& constraint) {
                      constraint->update(x);
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
std::tuple<bool, int, ResultType> AMSolver<ResultType, SolverArgsType>::solve(const SolverArgsType& args) {
    // Reset the cost and get rid of any carryover constraints from previous solves which need to be rebuilt
    resetCostMatrices();
    nonConstConstraints.clear();

    // Build new constraints and add to the cost matrices
    preSolve(args);

    // Ensure no carryover updates from previous solve
    resetConstraints();

    // Solve the problem
    auto [success, iters, result] = actualSolve(args); // TODO make explicit

    // Post-solve operations, e.g. modify the return format
    return std::make_tuple(success, iters, postSolve(result, args));
}

#endif // AMSOLVER_H