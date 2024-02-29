#ifndef AMSOLVER_H
#define AMSOLVER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <limits>

using namespace Eigen;

// Base class for constraints
class Constraint {
public:
    virtual ~Constraint() {}
    virtual SparseMatrix<double> getQuadCost(double rho) = 0;
    virtual VectorXd getLinearCost(double rho) = 0;
    virtual void update(double rho, const VectorXd& x) = 0;
    virtual void reset() = 0;
};

// Child class for equality constraints of form Gx = h
class EqualityConstraint : public Constraint {
private:
    SparseMatrix<double> G;
    VectorXd h;
    VectorXd lagrangeMult;

public:
    EqualityConstraint(const SparseMatrix<double>& G, const VectorXd& h);
    SparseMatrix<double> getQuadCost(double rho) const override;
    VectorXd getLinearCost(double rho) const override;
    void update(double rho, const VectorXd& x) override;
    void reset() override;
};

// Child class for inequality constraints of form Gx <= h
class InequalityConstraint : public Constraint {
private:
    SparseMatrix<double> G;
    VectorXd h;
    VectorXd slack;
    VectorXd lagrangeMult;

public:
    InequalityConstraint(const SparseMatrix<double>& G, const VectorXd& h);
    SparseMatrix<double> getQuadCost(double rho) const override;
    VectorXd getLinearCost(double rho) const override;
    void update(double rho, const VectorXd& x) override;
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

    VectorXd calculateOmega() const;
    VectorXd replicateVector(const VectorXd& vec, int times) const;

public:
    PolarInequalityConstraint(const SparseMatrix<double>& G, const VectorXd& c, double lwr_bound, double upr_bound, double bf_gamma);
    SparseMatrix<double> getQuadCost(double rho) const override;
    VectorXd getLinearCost(double rho) const override;
    void update(double rho, const VectorXd& x) override;
    void reset() override;
};

// Abstract class for solver
class AMSolver {
protected:
    std::vector<std::unique_ptr<Constraint>> constConstraints;
    std::vector<std::unique_ptr<Constraint>> nonConstConstraints;

    virtual void preSolve();
    virtual void postSolve();
    void actualSolve();

public:
    virtual ~AMSolver() {}
    void addConstraint(std::unique_ptr<Constraint> constraint, bool isConstant);
    void updateConstraints(double rho, const VectorXd& x);
    void solve() final;
};


#endif // AMSOLVER_H