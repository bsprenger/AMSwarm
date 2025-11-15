#include "constraint.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace amswarm {

using VectorXd = Eigen::VectorXd;
using SparseMatrixDouble = Eigen::SparseMatrix<double>;

EqualityConstraint::EqualityConstraint(const SparseMatrixDouble& G, const VectorXd& h,
                                       double tolerance)
    : G(G), G_T(G.transpose()), h(h), tolerance(tolerance) {
    if (G.rows() != h.size())
        throw std::invalid_argument("G and h are not compatible sizes");
    G_T_G = G_T * G;
    G_T_h = G_T * h;
}

const Eigen::SparseMatrix<double>& EqualityConstraint::getQuadraticTerm() const {
    return G_T_G;
}

VectorXd EqualityConstraint::getLinearTerm() const {
    return -G_T_h;
}

VectorXd EqualityConstraint::getBregmanUpdate(const VectorXd& x) const {
    if (G.cols() != x.size())
        throw std::invalid_argument("G and x are not compatible sizes");
    return G_T_G * x - G_T_h;  // See thesis document for derivation (Bregman iteration)
}

bool EqualityConstraint::isSatisfied(const VectorXd& x) const {
    return (G * x - h).cwiseAbs().maxCoeff() <= tolerance;  // Instead of maxcoeff could use .norm()
                                                            // < tolerance for stricter satisfaction
}

void EqualityConstraint::reset() {}

InequalityConstraint::InequalityConstraint(const SparseMatrixDouble& G, const VectorXd& h,
                                           double tolerance)
    : G(G), G_T(G.transpose()), h(h), tolerance(tolerance) {
    if (G.rows() != h.size())
        throw std::invalid_argument("G and h are not compatible sizes");
    slack = VectorXd::Zero(h.size());
    G_T_G = G_T * G;
    G_T_h = G_T * h;
}

const Eigen::SparseMatrix<double>& InequalityConstraint::getQuadraticTerm() const {
    return G_T_G;
}

VectorXd InequalityConstraint::getLinearTerm() const {
    return -G_T_h + G_T * slack;  // slack is added to the linear term (see thesis document for
                                  // derivation, converting inequality to equality constraint)
}

VectorXd InequalityConstraint::getBregmanUpdate(const VectorXd& x) const {
    if (G.cols() != x.size())
        throw std::invalid_argument("G and x are not compatible sizes");
    return G_T_G * x - G_T_h +
           G_T * slack;  // see thesis document for derivation (Bregman iteration)
}

void InequalityConstraint::update(const VectorXd& x) {
    if (G.cols() != x.size())
        throw std::invalid_argument("G and x are not compatible sizes");
    slack =
        (-G * x + h).array().max(0);  // Update the slack variables (see thesis document for
                                      // derivation, converting inequality to equality constraint)
}

bool InequalityConstraint::isSatisfied(const VectorXd& x) const {
    return (G * x - h).maxCoeff() < tolerance;  // Instead of maxcoeff could use .norm() < tolerance
                                                // for stricter satisfaction
}

void InequalityConstraint::reset() {
    slack.setZero();
}

PolarInequalityConstraint::PolarInequalityConstraint(const SparseMatrixDouble& G,
                                                     const VectorXd& c, double lwr_bound,
                                                     double upr_bound, double bf_gamma,
                                                     double tolerance)
    : G(G),
      G_T(G.transpose()),
      c(c),
      lwr_bound(lwr_bound),
      upr_bound(upr_bound),
      bf_gamma(bf_gamma),
      tolerance(tolerance) {
    if (G.rows() != c.size())
        throw std::invalid_argument("G and c are not compatible sizes");
    if (G.rows() % 3 != 0)
        throw std::invalid_argument("G must have a number of rows that is a multiple of 3");
    if (bf_gamma < 0 || bf_gamma > 1)
        throw std::invalid_argument("bf_gamma must be between 0 and 1");
    if (lwr_bound >= upr_bound)
        throw std::invalid_argument("lwr_bound must be strictly less than upr_bound");

    int n = c.size() / 3;
    h = -c;
    G_T_G = G_T * G;
    apply_upr_bound = !std::isinf(upr_bound);
    apply_lwr_bound = !std::isinf(lwr_bound);  // isinf returns true for negative infinity too
}

const Eigen::SparseMatrix<double>& PolarInequalityConstraint::getQuadraticTerm() const {
    return G_T_G;
}

VectorXd PolarInequalityConstraint::getLinearTerm() const {
    return -G_T * h;
}

VectorXd PolarInequalityConstraint::getBregmanUpdate(const VectorXd& x) const {
    if (G.cols() != x.size())
        throw std::invalid_argument("G and x are not compatible sizes");
    return G_T_G * x - G_T * h;  // See thesis document for derivation (Bregman iteration)
}

void PolarInequalityConstraint::update(const VectorXd& x) {
    // See thesis document for derivation of the following
    if (G.cols() != x.size())
        throw std::invalid_argument("G and x are not compatible sizes");

    VectorXd h_tmp = G * x + c;
    double prev_norm = 0;

    for (int i = 0; i < h_tmp.size(); i += 3) {
        // Calculate the norm of the current time segment
        double segment_norm = h_tmp.segment(i, 3).norm();
        double bound;

        // Apply the upper bound if it is not infinite
        if (apply_upr_bound) {
            if (i > 0) {
                bound = upr_bound - (1.0 - bf_gamma) * (upr_bound - prev_norm);
            } else {
                bound = upr_bound;
            }

            if (segment_norm > bound) {
                h_tmp.segment(i, 3) *= bound / segment_norm;
                segment_norm = bound;
            }
        }

        // Apply the lower bound if it is not infinite
        if (apply_lwr_bound) {
            if (i > 0) {
                bound = lwr_bound + (1.0 - bf_gamma) * (prev_norm - lwr_bound);
            } else {
                bound = lwr_bound;
            }

            if (segment_norm < bound) {
                h_tmp.segment(i, 3) *= bound / segment_norm;
                segment_norm = bound;
            }
        }

        // Keep track of the norm for the next iteration (for BF constraint)
        prev_norm = segment_norm;
    }
    h = h_tmp - c;
}

bool PolarInequalityConstraint::isSatisfied(const VectorXd& x) const {
    return (G * x - h).cwiseAbs().maxCoeff() < tolerance;
}

void PolarInequalityConstraint::reset() {
    h = -c;  // this essentially assumes initial guesses for angles/scaling are zero
}

} // namespace amswarm