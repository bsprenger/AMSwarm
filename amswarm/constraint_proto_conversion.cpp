#include "constraint_proto_conversion.h"
#include "constraint.pb.h"
#include <vector>

namespace amswarm {
namespace proto {

// Convert Eigen::SparseMatrix to protobuf
void toProto(const Eigen::SparseMatrix<double>& matrix, SparseMatrixProto* proto) {
    proto->set_rows(matrix.rows());
    proto->set_cols(matrix.cols());
    
    // Extract triplets from sparse matrix
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;
    
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            row_indices.push_back(it.row());
            col_indices.push_back(it.col());
            values.push_back(it.value());
        }
    }
    
    for (int idx : row_indices) {
        proto->add_row_indices(idx);
    }
    for (int idx : col_indices) {
        proto->add_col_indices(idx);
    }
    for (double val : values) {
        proto->add_values(val);
    }
}

// Convert protobuf to Eigen::SparseMatrix
void fromProto(const SparseMatrixProto& proto, Eigen::SparseMatrix<double>& matrix) {
    matrix.resize(proto.rows(), proto.cols());
    
    std::vector<Eigen::Triplet<double>> triplets;
    int nnz = proto.row_indices_size();
    
    for (int i = 0; i < nnz; ++i) {
        triplets.push_back(Eigen::Triplet<double>(
            proto.row_indices(i),
            proto.col_indices(i),
            proto.values(i)
        ));
    }
    
    matrix.setFromTriplets(triplets.begin(), triplets.end());
}

// Convert Eigen::VectorXd to protobuf
void toProto(const Eigen::VectorXd& vector, VectorProto* proto) {
    for (int i = 0; i < vector.size(); ++i) {
        proto->add_values(vector(i));
    }
}

// Convert protobuf to Eigen::VectorXd
void fromProto(const VectorProto& proto, Eigen::VectorXd& vector) {
    vector.resize(proto.values_size());
    for (int i = 0; i < proto.values_size(); ++i) {
        vector(i) = proto.values(i);
    }
}

// Convert EqualityConstraint to protobuf
void toProto(const EqualityConstraint& constraint, EqualityConstraintProto* proto) {
    toProto(constraint.getG(), proto->mutable_g());
    toProto(constraint.getH(), proto->mutable_h());
    proto->set_tolerance(constraint.getTolerance());
}

// Convert protobuf to EqualityConstraint
EqualityConstraint fromProto(const EqualityConstraintProto& proto) {
    Eigen::SparseMatrix<double> G;
    Eigen::VectorXd h;
    
    fromProto(proto.g(), G);
    fromProto(proto.h(), h);
    
    return EqualityConstraint(G, h, proto.tolerance());
}

// Convert InequalityConstraint to protobuf
void toProto(const InequalityConstraint& constraint, InequalityConstraintProto* proto) {
    toProto(constraint.getG(), proto->mutable_g());
    toProto(constraint.getH(), proto->mutable_h());
    toProto(constraint.getSlack(), proto->mutable_slack());
    proto->set_tolerance(constraint.getTolerance());
}

// Convert protobuf to InequalityConstraint
InequalityConstraint fromProto(const InequalityConstraintProto& proto) {
    Eigen::SparseMatrix<double> G;
    Eigen::VectorXd h;
    Eigen::VectorXd slack;
    
    fromProto(proto.g(), G);
    fromProto(proto.h(), h);
    fromProto(proto.slack(), slack);
    
    InequalityConstraint constraint(G, h, proto.tolerance());
    constraint.setSlack(slack);
    
    return constraint;
}

// Convert PolarInequalityConstraint to protobuf
void toProto(const PolarInequalityConstraint& constraint, PolarInequalityConstraintProto* proto) {
    toProto(constraint.getG(), proto->mutable_g());
    toProto(constraint.getC(), proto->mutable_c());
    toProto(constraint.getHInternal(), proto->mutable_h());
    proto->set_lwr_bound(constraint.getLowerBound());
    proto->set_upr_bound(constraint.getUpperBound());
    proto->set_bf_gamma(constraint.getBFGamma());
    proto->set_tolerance(constraint.getTolerance());
}

// Convert protobuf to PolarInequalityConstraint
PolarInequalityConstraint fromProto(const PolarInequalityConstraintProto& proto) {
    Eigen::SparseMatrix<double> G;
    Eigen::VectorXd c;
    Eigen::VectorXd h;
    
    fromProto(proto.g(), G);
    fromProto(proto.c(), c);
    fromProto(proto.h(), h);
    
    PolarInequalityConstraint constraint(
        G, c,
        proto.lwr_bound(),
        proto.upr_bound(),
        proto.bf_gamma(),
        proto.tolerance()
    );
    constraint.setHInternal(h);
    
    return constraint;
}

} // namespace proto
} // namespace amswarm
