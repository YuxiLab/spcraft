#include "eigeninterface.h"

#include <limits>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "core/CscMatrix.h"
#include "core/CsrMatrix.h"
#include "core/DenseVector.h"

namespace spcraft
{
namespace
{

using EigenCsc = Eigen::SparseMatrix<EigenInterfaceValue, Eigen::ColMajor, EigenInterfaceIndex>;
using EigenCsr = Eigen::SparseMatrix<EigenInterfaceValue, Eigen::RowMajor, EigenInterfaceIndex>;
using EigenVector = Eigen::Matrix<EigenInterfaceValue, Eigen::Dynamic, 1>;

Eigen::Map<const EigenCsc> ToEigen(const EigenInterfaceCsc& matrix)
{
  return Eigen::Map<const EigenCsc>(
      static_cast<Eigen::Index>(matrix.m), static_cast<Eigen::Index>(matrix.n),
      static_cast<Eigen::Index>(matrix.nnz), matrix.col_ptr, matrix.row_id, matrix.val, nullptr);
}

Eigen::Map<const EigenCsr> ToEigen(const EigenInterfaceCsr& matrix)
{
  return Eigen::Map<const EigenCsr>(
      static_cast<Eigen::Index>(matrix.m), static_cast<Eigen::Index>(matrix.n),
      static_cast<Eigen::Index>(matrix.nnz), matrix.row_ptr, matrix.col_id, matrix.val, nullptr);
}

Eigen::Map<const EigenVector> ToEigen(const EigenInterfaceVector& vector)
{
  return Eigen::Map<const EigenVector>(vector.val, static_cast<Eigen::Index>(vector.n));
}

}  // namespace

EigenComparison CompareSpMVWithEigen(const EigenInterfaceCsr& matrix,
                                     const EigenInterfaceVector& input,
                                     const EigenInterfaceVector& result,
                                     EigenInterfaceValue tolerance)
{
  if (matrix.n != input.n || matrix.m != result.n) {
    return {false, std::numeric_limits<double>::infinity()};
  }

  const EigenVector reference = ToEigen(matrix) * ToEigen(input);
  const auto actual = ToEigen(result);
  const bool matches = actual.isApprox(reference, tolerance);
  return {matches, matches ? 0.0 : (actual - reference).norm()};
}

EigenComparison CompareSpGEMMWithEigen(const EigenInterfaceCsc& lhs, const EigenInterfaceCsc& rhs,
                                       const EigenInterfaceCsc& result,
                                       EigenInterfaceValue tolerance)
{
  if (lhs.n != rhs.m || result.m != lhs.m || result.n != rhs.n) {
    return {false, std::numeric_limits<double>::infinity()};
  }

  const EigenCsc reference = ToEigen(lhs) * ToEigen(rhs);
  const auto actual = ToEigen(result);
  const bool matches = reference.isApprox(actual, tolerance);
  return {matches, matches ? 0.0 : (reference - actual).norm()};
}

}  // namespace spcraft
