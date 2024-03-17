#include "../include/constraint.h"

#include <stdexcept>
#include <cmath>
#include <limits>

using namespace Eigen;

inline double atan_approximation(double x) {
  constexpr double a1  =  0.99997726;
  constexpr double a3  = -0.33262347;
  constexpr double a5  =  0.19354346;
  constexpr double a7  = -0.11643287;
  constexpr double a9  =  0.05265332;
  constexpr double a11 = -0.01172120;

  double x_sq = x*x;
  return
    x * (a1 + x_sq * (a3 + x_sq * (a5 + x_sq * (a7 + x_sq * (a9 + x_sq * a11)))));
}

VectorXd atan2_eigen(const VectorXd& y, const VectorXd& x) {
    // Assuming y and x are the same size
    VectorXd result(y.size());

    // Custom unary operation, SIMD optimized
    auto atan2_op = [](const double y, const double x) -> double {
        if(std::isinf(x) && std::isinf(y)) return std::atan2(y, x);
        if(x == 0.0 && y == 0.0) return 0.0;

        bool swap = std::fabs(x) < std::fabs(y);
        double atan_input = (swap ? x : y) / (swap ? y : x);

        double res = atan_approximation(atan_input);
        res = swap ? (atan_input >= 0.0 ? M_PI_2 : -M_PI_2) - res : res;

        if      (x >= 0.0 && y >= 0.0) {}
        if (x < 0.0f) {
            res = (y >= 0.0f ? M_PI : -M_PI) + res;
        }
        else if (x >= 0.0 && y <  0.0) {}

        return res;
    };

    for (int i = 0; i < y.size(); ++i) {
        result(i) = atan2_op(y(i), x(i));
    }

    return result;
}

EqualityConstraint::EqualityConstraint(const SparseMatrix<double>& G, const VectorXd& h, double tolerance)
    : G(G), G_T(G.transpose()), h(h), tolerance(tolerance) {
    if (G.rows() != h.size()) throw std::invalid_argument("G and h are not compatible sizes");
    lagrangeMult = VectorXd::Zero(h.size());
    G_T_G = G_T * G;
}

SparseMatrix<double> EqualityConstraint::getQuadCost() const {
    return G_T_G;
}

VectorXd EqualityConstraint::getLinearCost(double rho) const {
    return -rho * G_T * (h - lagrangeMult / rho);
}

VectorXd EqualityConstraint::getBregmanUpdate(double rho, const VectorXd& x) const {
    if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
    return rho / 2 * G_T * (G * x - h);
}

void EqualityConstraint::update(double rho, const VectorXd& x) {
    if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
    if (useLagrange) {
        lagrangeMult += rho * (G * x - h);
    }
    // Note: For Bregman updates, AMSolver contains the multipliers
}

bool EqualityConstraint::isSatisfied(const VectorXd& x) const {
    return (G * x - h).cwiseAbs().maxCoeff() <= tolerance;
}

void EqualityConstraint::reset() {
    lagrangeMult.setZero();
}

InequalityConstraint::InequalityConstraint(const SparseMatrix<double>& G, const VectorXd& h, double tolerance)
    : G(G), G_T(G.transpose()), h(h), tolerance(tolerance) {
    if (G.rows() != h.size()) throw std::invalid_argument("G and h are not compatible sizes");
    slack = VectorXd::Zero(h.size());
    lagrangeMult = VectorXd::Zero(h.size());
    G_T_G = G_T * G;
}

SparseMatrix<double> InequalityConstraint::getQuadCost() const {
    return G_T_G;
}

VectorXd InequalityConstraint::getLinearCost(double rho) const {
    return -rho * G_T * (h - slack - lagrangeMult / rho);
}

VectorXd InequalityConstraint::getBregmanUpdate(double rho, const VectorXd& x) const {
    if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
    return rho / 2 * G_T * (G * x - h + slack);
}

void InequalityConstraint::update(double rho, const VectorXd& x) {
    if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
    VectorXd Gx_minus_h = G * x - h;
    slack = (- Gx_minus_h - lagrangeMult / rho).cwiseMax(0);
    if (useLagrange) {
        lagrangeMult += rho * (Gx_minus_h + slack);
    }
    // Note: For Bregman updates, AMSolver contains the multipliers
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
    : G(G), G_T(G.transpose()), c(c), lwr_bound(lwr_bound), upr_bound(upr_bound), bf_gamma(bf_gamma), tolerance(tolerance) {
    if (G.rows() != c.size()) throw std::invalid_argument("G and c are not compatible sizes");
    if (G.rows() % 3 != 0) throw std::invalid_argument("G must have a number of rows that is a multiple of 3");
    if (bf_gamma < 0 || bf_gamma > 1) throw std::invalid_argument("bf_gamma must be between 0 and 1");
    if (lwr_bound >= upr_bound) throw std::invalid_argument("lwr_bound must be strictly less than upr_bound");

    int n = c.size() / 3;
    alpha = VectorXd::Zero(n);
    beta = VectorXd::Zero(n);
    d = VectorXd::Zero(n);
    lagrangeMult = VectorXd::Zero(c.size());
    G_T_G = G_T * G;
}

SparseMatrix<double> PolarInequalityConstraint::getQuadCost() const {
    return G_T_G;
}

VectorXd PolarInequalityConstraint::getLinearCost(double rho) const {
    VectorXd omega = calculateOmega();
    VectorXd d_replicated = replicateVector(d, 3);
    VectorXd h = d_replicated.array() * omega.array() - c.array();
    return -rho * G_T * (h - lagrangeMult / rho);
}

VectorXd PolarInequalityConstraint::getBregmanUpdate(double rho, const VectorXd& x) const {
    if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
    VectorXd omega = calculateOmega();
    VectorXd d_replicated = replicateVector(d, 3);
    VectorXd h = d_replicated.array() * omega.array() - c.array();
    return rho / 2 * G_T * (G * x - h);
}

void PolarInequalityConstraint::update(double rho, const VectorXd& x) {
    if (G.cols() != x.size()) throw std::invalid_argument("G and x are not compatible sizes");
    
    // update alpha, beta, and d
    VectorXd constraint_vec = G * x + c + lagrangeMult / rho;
    // Prepare inputs for alpha and beta calculations
    int num_constraints = alpha.size(); // Assuming alpha and beta have the same size
    Eigen::VectorXd x_inputs_alpha(num_constraints);
    Eigen::VectorXd y_inputs_alpha(num_constraints);
    Eigen::VectorXd x_inputs_beta(num_constraints);
    Eigen::VectorXd y_inputs_beta(num_constraints);

    for (int i = 0; i < num_constraints; ++i) {
        double constraint_x = constraint_vec(i * 3);
        double constraint_y = constraint_vec(i * 3 + 1);
        double constraint_z = constraint_vec(i * 3 + 2);

        x_inputs_alpha(i) = constraint_x;
        y_inputs_alpha(i) = constraint_y;
        x_inputs_beta(i) = std::sqrt(constraint_x * constraint_x + constraint_y * constraint_y);
        y_inputs_beta(i) = constraint_z;
    }

    // Calculate alpha and beta using the atan2_eigen function
    alpha = atan2_eigen(y_inputs_alpha, x_inputs_alpha);
    beta = atan2_eigen(x_inputs_beta, y_inputs_beta);

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
    if (useLagrange) {
        lagrangeMult += (rho * ((G * x + c).array() - omega.array() * replicateVector(d, 3).array())).matrix();
    }
    // Note: For Bregman updates, AMSolver contains the multipliers
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