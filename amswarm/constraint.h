#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 * @file constraint.h
 * @brief Defines the base and derived classes for different types of constraints.
 *
 * Includes the abstract base class Constraint and derived classes for handling
 * equality, inequality, and polar inequality constraints. The derived classes
 * implement the virtual functions of the base class to provide the necessary
 * functionality for each type of constraint.
 */

namespace amswarm {

// Eigen type aliases
using VectorXd = Eigen::VectorXd;
template<typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;

/**
 * @class Constraint
 * @brief Abstract base class for constraints.
 *
 * Defines the interface for constraints and provides common functionality
 * for derived constraint classes.
 */
class Constraint {
public:
    Constraint() {}
    virtual ~Constraint() {}

    /**
     * Retrieves the quadratic term of the constraint-as-penalty.
     *
     * All constraints can be reformulated as quadratic penalties ||Ax - b||^2
     * in the cost function instead of hard constraints. This penalty can be
     * expanded into a quadratic term of the form x^T*Q*x, a linear term of the
     * form c^T * x, and a constant term. This function returns the sparse
     * matrix Q representing the quadratic term.
     * @return A const reference to a sparse matrix representing the quadratic term.
     */
    virtual const SparseMatrix<double>& getQuadraticTerm() const = 0;

    /**
     * Retrieves the linear term of the constraint-as-penalty.
     *
     * All constraints can be formulated as quadratic penalties ||Gx - h||^2
     * in the cost function instead of hard constraints. This penalty can be
     * expanded into a quadratic term of the form x^T*Q*x, a linear term of the
     * form c^T * x, and a constant term. This function returns the vector c
     * representing the linear term.
     * @return A vector representing the linear term.
     */
    virtual VectorXd getLinearTerm() const = 0;

    /**
     * Calculates and returns the Bregman "multiplier" update based on the current point.
     *
     * See thesis document for more information on Bregman iteration.
     * @param x The current point as a vector.
     * @return The Bregman update as a vector.
     */
    virtual VectorXd getBregmanUpdate(const VectorXd& x) const = 0;

    /**
     * Updates the internal state of the constraint if necessary (e.g. slack variables).
     * @param x The current point as a vector.
     */
    virtual void update(const VectorXd& x) {};

    /**
     * Checks if the constraint is satisfied at the given point.
     * @param x The point to check as a vector.
     * @return True if the constraint is satisfied, false otherwise.
     */
    virtual bool isSatisfied(const VectorXd& x) const = 0;

    /**
     * Resets the internal state of the constraint to its initial state.
     */
    virtual void reset() = 0;
};

/**
 * @class EqualityConstraint
 * @brief Handles equality constraints of the form Gx = h.
 */
class EqualityConstraint : public Constraint {
private:
    SparseMatrix<double> G;      // The matrix part of the constraint Gx = h
    VectorXd h;                  // The vector part of the constraint Gx = h
    SparseMatrix<double> G_T;    // Transpose of G, precomputed for efficiency
    SparseMatrix<double> G_T_G;  // G^T * G, precomputed for efficiency
    VectorXd G_T_h;              // G^T * h, precomputed for efficiency
    double tolerance;            // Tolerance within which the constraint is considered satisfied

public:
    EqualityConstraint(const SparseMatrix<double>& G, const VectorXd& h, double tolerance = 1e-2);
    const SparseMatrix<double>& getQuadraticTerm() const override;
    VectorXd getLinearTerm() const override;
    VectorXd getBregmanUpdate(const VectorXd& x) const override;
    bool isSatisfied(const VectorXd& x) const override;
    void reset() override;
};

/**
 * @class InequalityConstraint
 * @brief Handles inequality constraints of the form Gx <= h.
 */
class InequalityConstraint : public Constraint {
private:
    SparseMatrix<double> G;      // The matrix part of the constraint Gx <= h
    VectorXd h;                  // The vector part of the constraint Gx <= h
    SparseMatrix<double> G_T;    // Transpose of G, precomputed for efficiency
    SparseMatrix<double> G_T_G;  // G^T * G, precomputed for efficiency
    VectorXd G_T_h;              // G^T * h, precomputed for efficiency
    VectorXd
        slack;  // Slack variable to convert inequality to equality constraint (see thesis document)
    double tolerance;  // Tolerance within which the constraint is considered satisfied

public:
    InequalityConstraint(const SparseMatrix<double>& G, const VectorXd& h, double tolerance = 1e-2);
    const SparseMatrix<double>& getQuadraticTerm() const override;
    VectorXd getLinearTerm() const override;
    VectorXd getBregmanUpdate(const VectorXd& x) const override;
    void update(const VectorXd& x) override;
    bool isSatisfied(const VectorXd& x) const override;
    void reset() override;
};

/**
 * @class PolarInequalityConstraint
 * @brief Manages polar inequality constraints of a specialized form.
 *
 * This class is designed to handle constraints defined by polar coordinates that conform to the
 * formula: Gx + c = h(alpha, beta, d), with the boundary condition lwr_bound <= d <= upr_bound.
 *
 * Here, 'alpha', 'beta', and 'd' are vectors with a length of K+1, where 'd' represents the
 * distance from the origin, 'alpha' the azimuthal angle, and 'beta' the polar angle. The vector 'h'
 * has a length of 3(K+1), where each set of three elements in 'h()' represents a point in 3D space
 * expressed as:
 *
 * d[k] * [cos(alpha[k]) * sin(beta[k]), sin(alpha[k]) * sin(beta[k]), cos(beta[k])]^T
 *
 * This represents a unit vector defined by angles 'alpha[k]' and 'beta[k]', scaled by 'd[k]', where
 * 'k' is an index running from 0 to K. The index range from 0 to K can be interpreted as discrete
 * time steps, allowing this constraint to serve as a Barrier Function (BF) constraint to manage the
 * rate at which a constraint boundary is approached over successive time steps.
 */
class PolarInequalityConstraint : public Constraint {
private:
    SparseMatrix<double> G;      // The matrix part of the constraint Gx + c = h(alpha, beta, d)
    SparseMatrix<double> G_T;    // Transpose of G, precomputed for efficiency
    SparseMatrix<double> G_T_G;  // G^T * G, precomputed for efficiency
    VectorXd c;                  // The vector part of the constraint Gx + c = h(alpha, beta, d)
    VectorXd h;        // Variable to hold the h part of the constraint Gx + c = h(alpha, beta, d)
    double lwr_bound;  // can be -inf for unbounded (see -std::numeric_limits<double>::infinity())
    double upr_bound;  // can be +inf for unbounded (see std::numeric_limits<double>::infinity())
    bool apply_upr_bound;  // Flag to indicate if the upper bound is finite
    bool apply_lwr_bound;  // Flag to indicate if the lower bound is finite
    double bf_gamma;       // Barrier function gamma parameter (see thesis document)
    double tolerance;      // Tolerance within which the constraint is considered satisfied

public:
    PolarInequalityConstraint(const SparseMatrix<double>& G, const VectorXd& c, double lwr_bound,
                              double upr_bound, double bf_gamma = 1.0, double tolerance = 1e-2);
    const SparseMatrix<double>& getQuadraticTerm() const override;
    VectorXd getLinearTerm() const override;
    VectorXd getBregmanUpdate(const VectorXd& x) const override;
    void update(const VectorXd& x) override;
    bool isSatisfied(const VectorXd& x) const override;
    void reset() override;
};

} // namespace amswarm

#endif  // CONSTRAINT_H