#include "../include/constraint.h"

#include <stdexcept>
#include <cmath>
#include <limits>

using namespace Eigen;

EqualityConstraint::EqualityConstraint(const SparseMatrix<double>& G, const VectorXd& h, double tolerance)
    : G(G), h(h), tolerance(tolerance) {
    if (G.rows() != h.size()) throw std::invalid_argument("G and h are not compatible sizes");
    lagrangeMult = VectorXd::Zero(h.size());
}

SparseMatrix<double> EqualityConstraint::getQuadCost(double rho) const {
    return rho * G.transpose() * G;
}

VectorXd EqualityConstraint::getLinearCost(double rho) const {
    return -rho * G.transpose() * (h - lagrangeMult / rho);
}

void EqualityConstraint::update(double rho, const VectorXd& x) {
    if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
    lagrangeMult += rho * (G * x - h);
}

bool EqualityConstraint::isSatisfied(const VectorXd& x) const {
    return (G * x - h).cwiseAbs().maxCoeff() <= tolerance;
}

void EqualityConstraint::reset() {
    lagrangeMult.setZero();
}

InequalityConstraint::InequalityConstraint(const SparseMatrix<double>& G, const VectorXd& h, double tolerance)
    : G(G), h(h), tolerance(tolerance) {
    if (G.rows() != h.size()) throw std::invalid_argument("G and h are not compatible sizes");
    slack = VectorXd::Zero(h.size());
    lagrangeMult = VectorXd::Zero(h.size());
}

SparseMatrix<double> InequalityConstraint::getQuadCost(double rho) const {
    return rho * G.transpose() * G;
}

VectorXd InequalityConstraint::getLinearCost(double rho) const {
    return -rho * G.transpose() * (h - slack - lagrangeMult / rho);
}

void InequalityConstraint::update(double rho, const VectorXd& x) {
    if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
    slack = (- G * x + h - lagrangeMult / rho).cwiseMax(0);
    lagrangeMult += rho * (G * x - h + slack);
}

bool InequalityConstraint::isSatisfied(const VectorXd& x) const {
    return (G * x - h).maxCoeff() < tolerance;
}

void InequalityConstraint::reset() {
    slack.setZero();
    lagrangeMult.setZero();
}

VectorXd PolarInequalityConstraint::calculateOmega() const {
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

VectorXd PolarInequalityConstraint::replicateVector(const VectorXd& vec, int times) const {
    VectorXd replicated(vec.size() * times);
    for (int i = 0; i < vec.size(); ++i) {
        replicated.segment(i * times, times).setConstant(vec(i)); // TODO check if fixed-size is faster
    }
    return replicated;
}

PolarInequalityConstraint::PolarInequalityConstraint(const SparseMatrix<double>& G, const VectorXd& c, double lwr_bound, double upr_bound, double bf_gamma, double tolerance)
    : G(G), c(c), lwr_bound(lwr_bound), upr_bound(upr_bound), bf_gamma(bf_gamma), tolerance(tolerance) {
    if (G.rows() != c.size()) throw std::invalid_argument("G and c are not compatible sizes");
    if (G.rows() % 3 != 0) throw std::invalid_argument("G must have a number of rows that is a multiple of 3");
    if (bf_gamma < 0 || bf_gamma > 1) throw std::invalid_argument("bf_gamma must be between 0 and 1");
    if (lwr_bound >= upr_bound) throw std::invalid_argument("lwr_bound must be strictly less than upr_bound");

    int n = c.size() / 3;
    alpha = VectorXd::Zero(n);
    beta = VectorXd::Zero(n);
    d = VectorXd::Zero(n);
    lagrangeMult = VectorXd::Zero(c.size());
}

SparseMatrix<double> PolarInequalityConstraint::getQuadCost(double rho) const {
    return rho * G.transpose() * G;
}

VectorXd PolarInequalityConstraint::getLinearCost(double rho) const {
    VectorXd omega = calculateOmega();
    VectorXd d_replicated = replicateVector(d, 3);
    VectorXd h = d_replicated.array() * omega.array() - c.array();
    return -rho * G.transpose() * (h - lagrangeMult / rho);
}

void PolarInequalityConstraint::update(double rho, const VectorXd& x) {
    if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
    
    // update alpha, beta, and d
    VectorXd constraint_vec = G * x + c + lagrangeMult / rho;
    for (int i = 0; i < alpha.size(); ++i) {
        double constraint_x = constraint_vec(i * 3);
        double constraint_y = constraint_vec(i * 3 + 1);
        double constraint_z = constraint_vec(i * 3 + 2);

        alpha(i) = std::atan2(constraint_y, constraint_x);
        beta(i) = std::atan2(std::sqrt(constraint_x*constraint_x + constraint_y*constraint_y), constraint_z);
    }

    // update d
    VectorXd omega = calculateOmega();

    // Precompute adjustments based on bounds
    bool apply_upr_bound = !std::isinf(upr_bound);
    bool apply_lwr_bound = !std::isinf(lwr_bound); // isinf returns true for negative infinity too
    for (int i = 0; i < d.size(); ++i) {
        d(i) = constraint_vec.segment(i * 3, 3).dot(omega.segment(i * 3, 3));
        if (i > 0) {
            if (apply_upr_bound) {
                d(i) = std::min(d(i), upr_bound - (1.0 - bf_gamma)*(upr_bound - d(i-1)));
            }
            if (apply_lwr_bound) {
                d(i) = std::max(d(i), lwr_bound + (1.0 - bf_gamma)*(d(i-1) - lwr_bound));
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

    lagrangeMult += (rho * ((G * x + c).array() - omega.array() * replicateVector(d, 3).array())).matrix();
}

bool PolarInequalityConstraint::isSatisfied(const VectorXd& x) const {
    return ((G * x + c).array() - calculateOmega().array() * replicateVector(d, 3).array()).cwiseAbs().maxCoeff() < tolerance;
}

void PolarInequalityConstraint::reset() {
    alpha.setZero();
    beta.setZero();
    d.setZero();
    lagrangeMult.setZero();
}