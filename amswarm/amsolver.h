#ifndef AMSOLVER_H
#define AMSOLVER_H

#include "constraint.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <vector>
#include <memory>
#include <algorithm>
#include <execution>

/**
 * @file AMSolver.h
 * @brief Header file defining the AMSolver class and associated configuration structure.
 *
 * This file contains the definition of the AMSolver class template, which is designed for solving optimization
 * problems using the Alternating Minimization (AM) algorithm. It includes a configuration structure, AMSolverConfig,
 * for setting up solver parameters such as the initial penalty, maximum penalty, and maximum iterations. The AMSolver
 * class is an abstract template that requires specification of the result type and the solver argument types, making
 * it customizable for various optimization problems.
 * 
 * Child classes should inherit from the AMSolver class and implement the preSolve and postSolve functions to define the
 * optimization problem and the post-processing of the solution, respectively. The AMSolver class provides a generic
 * solve function that orchestrates the solving process, including pre-solve setup, the actual solving process, and
 * post-solve processing. Child classes should not override the solve or actualSolve functions (which define the
 * actual algorithm and workflow), but can customize the optimization problem by implementing the preSolve and
 * postSolve functions.
 *
 * The main entry point for the user is the solve function, which should be called with the appropriate arguments
 * to solve the optimization problem from a child class.
 */

using namespace Eigen;

/**
 * @struct AMSolverConfig
 * @brief Configuration settings for the AMSolver.
 * 
 * Encapsulates configuration options for the AMSolver, including parameters like the initial penalty parameter (`rho_init`),
 * the maximum allowed penalty parameter (`max_rho`), and the maximum number of iterations to run the solver (`max_iters`).
 */
struct AMSolverConfig {
    double rho_init;
    double max_rho;
    int max_iters;

    // Default constructor
    AMSolverConfig () {};

    /**
     * @brief Constructor with initialization.
     * @param rho_init Initial value of rho.
     * @param max_rho Maximum allowable value of rho.
     * @param max_iters Maximum number of iterations.
     */
    AMSolverConfig(double rho_init, double max_rho, int max_iters)
        : rho_init(rho_init), max_rho(max_rho), max_iters(max_iters) {};
};

/**
 * @class AMSolver
 * @brief Abstract class template for an Alternating Minimization (AM) Solver.
 * 
 * The AMSolver is designed to solve optimization problems by alternating between solving for the optimization variables and updating 
 * dual variables, using a specified set of constraints. The solver is customizable, allowing for various types
 * of constraints and optimization problems to be solved. The idea is to allow for creating custom solvers by inheriting from this class
 * and implementing the preSolve and postSolve functions.
 * 
 * @tparam ResultType The type of the result produced by the solver.
 * @tparam SolverArgsType The type of the arguments required for solving the problem.
 */
template<typename ResultType, typename SolverArgsType>
class AMSolver {
protected:
    std::vector<std::unique_ptr<Constraint>> constConstraints; // Constant constraints that do not change between solves. They are known at initialization of the solver.
    std::vector<std::unique_ptr<Constraint>> nonConstConstraints; // Non-constant constraints that can be updated between solves.
    AMSolverConfig solverConfig; // Configuration for the solver.

    // cost of the form 0.5 * x^T * quadCost * x + x^T * linearCost
    SparseMatrix<double> initialQuadCost; // The initial quadratic cost matrix which can be set at construction by child classes.
    VectorXd initialLinearCost; // The initial linear cost vector which can be set at construction by child classes.
    SparseMatrix<double> quadCost; // The current quadratic cost matrix (is modified during solving)
    VectorXd linearCost; // The current linear cost vector (is modified during solving)

    /**
     * @brief Called before the solve process begins, allowing for setup of the optimization problem
     * and the constraints which change at solve time (adding non-const constraints, modifying the cost matrices, etc.)
     * This function should be implemented by child classes for the specific optimization problem.
     * @param args The arguments required for the pre-solve process.
     */
    virtual void preSolve(const SolverArgsType& args) = 0;
    
    /**
     * @brief Called after the solve process ends, allowing for any necessary cleanup or result processing.
     * This function should be implemented by child classes for the specific optimization problem.
     * @param x The solution vector from the solve process.
     * @param args The arguments required for the post-solve process.
     * @return The result of the post-solve process, of type ResultType.
     */
    virtual ResultType postSolve(const VectorXd& x, const SolverArgsType& args) = 0;

    /**
     * @brief Conducts the actual solving process, implementing the optimization algorithm.
     * This function is NOT meant to be overridden by child classes, as it contains the core solving logic.
     * @param args The arguments required for solving the problem.
     * @return A tuple containing a success flag, the number of iterations, and the solution vector.
     */
    std::tuple<bool, int, VectorXd> actualSolve(const SolverArgsType& args);

    /// @brief Resets the cost matrices to their initial values.
    void resetCostMatrices();

public:
    AMSolver(AMSolverConfig config) : solverConfig(config) {};
    virtual ~AMSolver() = default;

    /**
     * @brief Adds a constraint to the solver via a unique pointer
     * (which transfers ownership to the solver, as constraints should not be
     * modified by multiple solvers at once).
     * @param constraint The constraint to be added.
     * @param isConstant Indicates if the constraint is constant (true) or can be updated (false).
     */
    void addConstraint(std::unique_ptr<Constraint> constraint, bool isConstant);

    /**
     * @brief Updates all the non-constant constraints based on the current optimization variables.
     * @param x The current optimization variables.
     */
    void updateConstraints(const VectorXd& x);

    /// @brief Resets the constraints to their initial state.
    void resetConstraints();

    /**
     * @brief Solves the optimization problem. This is the main function to be called by the user and should NOT be overridden as it
     * contains the main solving workflow (preSolve, actualSolve, postSolve).
     * @param args The arguments required for solving the problem.
     * @return A tuple containing a success flag, the number of iterations, and the result of type ResultType.
     */
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
    SimplicialLDLT<SparseMatrix<double>> linearSolver; // SimplicialLDLT is chosen for its efficiency in dealing with sparse systems.

    int iters  = 0;
    double rho = solverConfig.rho_init;
    bool solver_initialized = false;

    SparseMatrix<double> Q(quadCost.rows(), quadCost.cols()); // Holds the combined quadratic terms
    VectorXd q = VectorXd::Zero(quadCost.rows()); // Holds the combined linear terms
    VectorXd x = VectorXd::Zero(quadCost.rows()); // The optimization variable
    VectorXd bregmanMult = VectorXd::Zero(quadCost.rows()); // The Bregman multiplier (see thesis document for details on Bregman iteration)

    // Aggregate the quadratic and linear terms from all constraints.
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

    // Iteratively solve the optimization problem until a solution is found or the iteration limit is reached.
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

        // With the new solution x, recalculate the linear term so we can get the Bregman multiplier
        linearConstraintTerms.setZero();
        for (auto& constraint : constConstraints) {
            linearConstraintTerms += constraint->getLinearTerm();
        }
        for (auto& constraint : nonConstConstraints) {
            linearConstraintTerms += constraint->getLinearTerm();
        }

        // Calculate the Bregman multiplier (see thesis document for derivation)
        VectorXd bregmanUpdate = 0.5 * (quadConstraintTerms * x + linearConstraintTerms);
        bregmanMult -= bregmanUpdate;
        
        // Gradually increase the penalty parameter to enforce constraints
        rho *= solverConfig.rho_init;
        rho = std::min(rho, solverConfig.max_rho);
        iters++;
    }

    return std::make_tuple(false, iters, x); // Indicate failure with the bool. Still return the vector anyways
}

template<typename ResultType, typename SolverArgsType>
void AMSolver<ResultType, SolverArgsType>::addConstraint(std::unique_ptr<Constraint> constraint, bool isConstant) {
    // Add a constraint to the appropriate list, based on whether it is constant or not.
    // This separation allows for efficient management and updates of constraints between solves.
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
    // Resets all constraints to their initial states. This is crucial when re-using the solver instance
    // for multiple solves, ensuring that stale state from previous solves does not affect new ones.
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
    // The main entry point for solving the optimization problem. This method orchestrates the entire
    // process, from preparation through to obtaining and processing the final solution.
    // This is the function that should be called by the user to solve the problem. To customize the
    // solving process, the preSolve and postSolve functions should be implemented in child classes, but
    // this function should not be overridden, and neither should the actualSolve function.

    // Reset the cost and get rid of any carryover constraints from previous solves which need to be rebuilt
    resetCostMatrices();
    nonConstConstraints.clear();

    // Build new constraints and add to the cost matrices
    preSolve(args); // Custom pre-solve logic, implemented by derived classes.

    // Ensure no carryover updates from previous solve
    resetConstraints();

    // Execute the solve process, obtaining the raw solution vector.
    auto [success, iters, result] = actualSolve(args);

    // Post-processing of the solution, as defined by derived classes, e.g. modify the return format
    return std::make_tuple(success, iters, postSolve(result, args));
}

#endif // AMSOLVER_H