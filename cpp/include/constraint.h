#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

// Base class for constraints
class Constraint {
public:
    virtual ~Constraint() {}
    virtual SparseMatrix<double> getQuadCost(double rho) const = 0;
    virtual VectorXd getLinearCost(double rho) const = 0;
    virtual void update(double rho, const VectorXd& x) = 0;
    virtual bool isSatisfied(const VectorXd& x) const = 0;
    virtual void reset() = 0;
};

// Child class for equality constraints of form Gx = h
class EqualityConstraint : public Constraint {
private:
    SparseMatrix<double> G;
    VectorXd h;
    VectorXd lagrangeMult;
    double tolerance;

public:
    EqualityConstraint(const SparseMatrix<double>& G, const VectorXd& h, double tolerance = 1e-3);
    SparseMatrix<double> getQuadCost(double rho) const override;
    VectorXd getLinearCost(double rho) const override;
    void update(double rho, const VectorXd& x) override;
    bool isSatisfied(const VectorXd& x) const override;
    void reset() override;
};

// Child class for inequality constraints of form Gx <= h
class InequalityConstraint : public Constraint {
private:
    SparseMatrix<double> G;
    VectorXd h;
    VectorXd slack;
    VectorXd lagrangeMult;
    double tolerance;

public:
    InequalityConstraint(const SparseMatrix<double>& G, const VectorXd& h, double tolerance = 1e-3);
    SparseMatrix<double> getQuadCost(double rho) const override;
    VectorXd getLinearCost(double rho) const override;
    void update(double rho, const VectorXd& x) override;
    bool isSatisfied(const VectorXd& x) const override;
    void reset() override;
};

// Child class for polar inequality constraints of form {Gx + c = h(alpha, beta, d), d <= upr_bound}
class PolarInequalityConstraint : public Constraint {
private:
    SparseMatrix<double> G;
    VectorXd c;
    VectorXd alpha;
    VectorXd beta;
    VectorXd d;
    VectorXd lagrangeMult;
    double lwr_bound; // can be -inf for unbounded
    double upr_bound; // can be +inf for unbounded
    double bf_gamma;
    double tolerance;

    VectorXd calculateOmega() const;
    VectorXd replicateVector(const VectorXd& vec, int times) const;

public:
    PolarInequalityConstraint(const SparseMatrix<double>& G, const VectorXd& c, double lwr_bound, double upr_bound, double bf_gamma = 1.0, double tolerance = 1e-3);
    SparseMatrix<double> getQuadCost(double rho) const override;
    VectorXd getLinearCost(double rho) const override;
    void update(double rho, const VectorXd& x) override;
    bool isSatisfied(const VectorXd& x) const override;
    void reset() override;
};

#endif // CONSTRAINT_H