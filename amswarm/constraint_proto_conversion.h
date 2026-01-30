#ifndef CONSTRAINT_PROTO_CONVERSION_H
#define CONSTRAINT_PROTO_CONVERSION_H

#include "constraint.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Forward declarations of protobuf message types
namespace amswarm {
    class SparseMatrixProto;
    class VectorProto;
    class EqualityConstraintProto;
    class InequalityConstraintProto;
    class PolarInequalityConstraintProto;
}

namespace amswarm {

/**
 * @brief Conversion utilities for serializing constraint classes to protobuf and back.
 * 
 * This namespace provides functions to convert between C++ constraint classes and their
 * protobuf representations, enabling serialization and deserialization of constraint objects.
 */
namespace proto {

// Helper functions for Eigen types
void toProto(const Eigen::SparseMatrix<double>& matrix, SparseMatrixProto* proto);
void fromProto(const SparseMatrixProto& proto, Eigen::SparseMatrix<double>& matrix);

void toProto(const Eigen::VectorXd& vector, VectorProto* proto);
void fromProto(const VectorProto& proto, Eigen::VectorXd& vector);

// EqualityConstraint conversions
void toProto(const EqualityConstraint& constraint, EqualityConstraintProto* proto);
EqualityConstraint fromProto(const EqualityConstraintProto& proto);

// InequalityConstraint conversions
void toProto(const InequalityConstraint& constraint, InequalityConstraintProto* proto);
InequalityConstraint fromProto(const InequalityConstraintProto& proto);

// PolarInequalityConstraint conversions
void toProto(const PolarInequalityConstraint& constraint, PolarInequalityConstraintProto* proto);
PolarInequalityConstraint fromProto(const PolarInequalityConstraintProto& proto);

} // namespace proto
} // namespace amswarm

#endif  // CONSTRAINT_PROTO_CONVERSION_H
