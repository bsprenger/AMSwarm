#include <amsolver.h>

using namespace Eigen;


EqualityConstraint::EqualityConstraint(const SparseMatrix<double>& G, const VectorXd& h) : G(G), h(h) {
    if (G.rows() != h.size()) {
        throw std::invalid_argument("G and h are not compatible sizes");
    }
    lagrangeMult = VectorXd::Zero(h.size());
}

SparseMatrix<double> EqualityConstraint::getQuadCost(double rho) {
    return rho * G.transpose() * G;
}

VectorXd EqualityConstraint::getLinearCost(double rho) {
    return -2 * rho * G.transpose() * (h - lagrangeMult / rho);
}

void EqualityConstraint::update(double rho, const VectorXd& x) {
    if (G.cols() != x.size()) {
        throw std::invalid_argument("G and x are not compatible sizes");
    }
    lagrangeMult += rho * (G * x - h);
}

void EqualityConstraint::reset() {
    lagrangeMult.setZero();
}

InequalityConstraint::InequalityConstraint(const SparseMatrix<double>& G, const VectorXd& h) : G(G), h(h) {
    if (G.rows() != h.size()) {
        throw std::invalid_argument("G and h are not compatible sizes");
    }
    slack = VectorXd::Zero(h.size());
    lagrangeMult = VectorXd::Zero(h.size());
}

SparseMatrix<double> InequalityConstraint::getQuadCost(double rho) {
    return rho * G.transpose() * G;
}

VectorXd InequalityConstraint::getLinearCost(double rho) {
    return -2 * rho * G.transpose() * (h - slack - lagrangeMult / rho);
}

void InequalityConstraint::update(double rho, const VectorXd& x) {
    if (G.cols() != x.size()) {
        throw std::invalid_argument("G and x are not compatible sizes");
    }
    slack = (- G * x + h - lagrangeMult / rho).cwiseMax(0);
    lagrangeMult += rho * (G * x - h - slack);
}

void InequalityConstraint::reset() {
    slack.setZero();
    lagrangeMult.setZero();
}