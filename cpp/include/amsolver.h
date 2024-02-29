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
    SparseMatrix<double> getQuadCost(double rho) override;
    VectorXd getLinearCost(double rho) override;
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
    SparseMatrix<double> getQuadCost(double rho) override;
    VectorXd getLinearCost(double rho) override;
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

    VectorXd calculateOmega() const {
        VectorXd cos_alpha = alpha.array().cos();
        VectorXd sin_alpha = alpha.array().sin();
        VectorXd cos_beta = beta.array().cos();
        VectorXd sin_beta = beta.array().sin();
        
        VectorXd omega(3 * alpha.size());
        for (int i = 0; i < alpha.size(); ++i) {
            omega.segment<3>(i * 3) << cos_alpha(i) * sin_beta(i), sin_alpha(i) * sin_beta(i), cos_beta(i);
        }
        return omega;
    }

    VectorXd replicateVector(const VectorXd& vec, int times) const {
        VectorXd replicated(vec.size() * times);
        for (int i = 0; i < vec.size(); ++i) {
            replicated.segment(i * times, times).setConstant(vec(i));
        }
        return replicated;
    }

public:
    PolarInequalityConstraint(const SparseMatrix<double>& G, const VectorXd& c, double lwr_bound, double upr_bound, double bf_gamma)
        : G(G), c(c), lwr_bound(lwr_bound), upr_bound(upr_bound), bf_gamma(bf_gamma) {
        if (G.rows() != c.size()) throw std::invalid_argument("G and c are not compatible sizes");
        if (bf_gamma < 0 || bf_gamma > 1) throw std::invalid_argument("bf_gamma must be between 0 and 1");
        if (lwr_bound >= upr_bound) throw std::invalid_argument("lwr_bound must be strictly less than upr_bound");

        int n = c.size() / 3;
        alpha = VectorXd::Zero(n);
        beta = VectorXd::Zero(n);
        d = VectorXd::Zero(n);
        lagrangeMult = VectorXd::Zero(c.size());
    }

    SparseMatrix<double> getQuadCost(double rho) {
        return rho * G.transpose() * G;
    }

    VectorXd getLinearCost(double rho) {
        VectorXd omega = calculateOmega();
        VectorXd d_replicated = replicateVector(d, 3);
        VectorXd h = d_replicated.array() * omega.array() - c.array();
        return -2 * rho * G.transpose() * (h - lagrangeMult / rho);
    }

    void update(double rho, const VectorXd& x) {
        if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
        
        // update alpha, beta, and d
        VectorXd constraint_vec = G * x + c + lagrangeMult / rho;
        for (int i = 0; i < alpha.size(); ++i) {
            double constraint_x = constraint_vec(i * 3);
            double constraint_y = constraint_vec(i * 3 + 1);
            double constraint_z = constraint_vec(i * 3 + 2);

            alpha(i) = std::atan2(constraint_y, constraint_x);
            beta(i) = std::atan2(constraint_x, constraint_z);
        }

        // update d
        VectorXd omega = calculateOmega();

        // Precompute adjustments based on bounds
        bool apply_upr_bound = !std::isinf(upr_bound);
        bool apply_lwr_bound = !std::isinf(lwr_bound);

        for (int i = 0; i < d.size(); ++i) {
            d(i) = constraint_vec.segment(i * 3, 3).dot(omega.segment(i * 3, 3));
            if (i > 0) {
                if (apply_upr_bound) {
                    d(i) = std::min(d(i), upr_bound + (1.0 - bf_gamma)*(d(i-1) - upr_bound)); // TODO check this
                }
                if (apply_lwr_bound) {
                    d(i) = std::max(d(i), lwr_bound + (1.0 - bf_gamma)*(d(i-1) - lwr_bound)); // TODO check this
                }
            } else {
                if (apply_upr_bound) {
                    d(i) = std::min(d(i), upr_bound);
                }
                if (apply_lwr_bound) {
                    d(i) = std::max(d(i), lwr_bound);
                }
            }
        }

        lagrangeMult += rho * (G * x - omega.array() * replicateVector(d, 3).array() - c.array()); // TODO check this
    }

    void reset() {
        alpha.setZero();
        beta.setZero();
        d.setZero();
        lagrangeMult.setZero();
    }
};

// Abstract class for solver
class AMSolver {
protected:
    std::vector<std::unique_ptr<Constraint>> constConstraints;
    std::vector<std::unique_ptr<Constraint>> nonConstConstraints;

    virtual void preSolve() {} // skeleton, must be overriden
    virtual void postSolve() {} // skeleton, must be overriden

    void actualSolve() {
        SimplicialLDLT<SparseMatrix<double>> linearSolver;

        int iters  = 0;
        double rho = 1.3;
        int max_iters = 1000;
        bool solver_initialized = false;

        while (iters < max_iters) {
            // Construct the quadratic and linear cost matrices
            SparseMatrix<double> quadCost = SparseMatrix<double>(0, 0); // TODO check this
            VectorXd linearCost = VectorXd::Zero(0); // TODO check this
            for (auto& constraint : constConstraints) {
                quadCost += constraint->getQuadCost(rho);
                linearCost += constraint->getLinearCost(rho);
            }
            for (auto& constraint : nonConstConstraints) {
                quadCost += constraint->getQuadCost(rho);
                linearCost += constraint->getLinearCost(rho);
            }

            // Solve the linear system
            if (!solver_initialized) {
                linearSolver.analyzePattern(quadCost);
                solver_initialized = true;
            }
            solver.factorize(quadCost);
            zeta_1 = solver.solve(-linearCost); // TODO check this
            
            // Update the constraints
            updateConstraints(rho, ????); // TODO

            // Check for convergence

            // Update the penalty parameter
            rho *= 10; // TODO proper update
            iters++;
        }
    }

public:
    virtual ~AMSolver() {}

    void addConstraint(std::unique_ptr<Constraint> constraint, bool isConstant) {
        if (isConstant) {
            constConstraints.push_back(std::move(constraint));
        } else {
            nonConstConstraints.push_back(std::move(constraint));
        }
    }

    void updateConstraints(double rho, const VectorXd& x) {
        for (auto& constraint : constConstraints) {
            constraint->update(rho, x);
        }
        for (auto& constraint : nonConstConstraints) {
            constraint->update(rho, x);
        }
    }

    void solve() final {
        preSolve();
        actualSolve();
        postSolve();
    }
};


#endif // AMSOLVER_H