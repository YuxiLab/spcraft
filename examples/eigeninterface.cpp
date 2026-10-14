#include "eigeninterface.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "core/CscMatrix.h"
#include "core/CsrMatrix.h"
#include "core/DenseVector.h"

namespace spcraft
{
namespace
{

template <class NT>
using EigenCsc = Eigen::SparseMatrix<NT, Eigen::ColMajor, std::int64_t>;

template <class NT>
using EigenCsr = Eigen::SparseMatrix<NT, Eigen::RowMajor, std::int64_t>;

template <class NT>
using EigenVector = Eigen::Matrix<NT, Eigen::Dynamic, 1>;

SP_MAT_TEMP
EigenCsc<NT> ToEigen(const SPCSC& matrix)
{
  // Eigen uses one storage-index type for both indices and offsets. Triplets
  // support SpCraft's mixed IT/OT pairs without narrowing the stored offsets.
  std::vector<Eigen::Triplet<NT, std::int64_t>> entries;
  entries.reserve(static_cast<std::size_t>(matrix.nnz));
  for (IT col = 0; col < matrix.n; ++col) {
    for (OT p = matrix.col_ptr[col]; p < matrix.col_ptr[col + 1]; ++p) {
      entries.emplace_back(matrix.row_id[p], col, matrix.val[p]);
    }
  }
  EigenCsc<NT> result(matrix.m, matrix.n);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

SP_MAT_TEMP
EigenCsr<NT> ToEigen(const SPCSR& matrix)
{
  std::vector<Eigen::Triplet<NT, std::int64_t>> entries;
  entries.reserve(static_cast<std::size_t>(matrix.nnz));
  for (IT row = 0; row < matrix.m; ++row) {
    for (OT p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p) {
      entries.emplace_back(row, matrix.col_id[p], matrix.val[p]);
    }
  }
  EigenCsr<NT> result(matrix.m, matrix.n);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

SP_VEC_TEMP
Eigen::Map<const EigenVector<NT>> ToEigen(const DenseVector<IT, NT>& vector)
{
  Eigen::Map<const EigenVector<NT>> result(vector.val, static_cast<Eigen::Index>(vector.n));
  return result;
}

}  // namespace

SP_MAT_TEMP
EigenComparison CompareSpMVWithEigen(const SPCSR& a, const SPDVEC& x, const SPDVEC& y, NT tol)
{
  if (a.n != x.n || a.m != y.n) {
    return {false, std::numeric_limits<double>::infinity()};
  }

  const auto eigen_matrix = ToEigen(a);
  const auto eigen_input = ToEigen(x);
  const auto start = std::chrono::steady_clock::now();
  const EigenVector<NT> reference = eigen_matrix * eigen_input;
  const auto stop = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(stop - start).count();
  const auto actual = ToEigen(y);
  const bool matches = actual.isApprox(reference, tol);
  return {matches, matches ? 0.0 : (actual - reference).norm(), seconds};
}

SP_MAT_TEMP
EigenComparison CompareSpGEMMWithEigen(const SPCSC& a, const SPCSC& b, const SPCSC& c, NT tol)
{
  if (a.n != b.m || c.m != a.m || c.n != b.n) {
    return {false, std::numeric_limits<double>::infinity()};
  }

  const auto eigen_lhs = ToEigen(a);
  const auto eigen_rhs = ToEigen(b);
  const auto start = std::chrono::steady_clock::now();
  const EigenCsc<NT> reference = eigen_lhs * eigen_rhs;
  const auto stop = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(stop - start).count();
  const auto actual = ToEigen(c);
  const bool matches = reference.isApprox(actual, tol);
  return {matches, matches ? 0.0 : (reference - actual).norm(), seconds};
}

#define SPCRAFT_DEFINE_EIGEN_REFERENCE_TYPES(IT, NT, OT)                                         \
  template EigenComparison CompareSpMVWithEigen<IT, NT, OT>(                                     \
      const CsrMatrix<IT, NT, OT>&, const DenseVector<IT, NT>&, const DenseVector<IT, NT>&, NT); \
  template EigenComparison CompareSpGEMMWithEigen<IT, NT, OT>(const CscMatrix<IT, NT, OT>&,      \
                                                              const CscMatrix<IT, NT, OT>&,      \
                                                              const CscMatrix<IT, NT, OT>&, NT);

#define SPCRAFT_DEFINE_EIGEN_REFERENCE(r, product)                                          \
  SPCRAFT_DEFINE_EIGEN_REFERENCE_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                       BOOST_PP_SEQ_ELEM(1, product),                       \
                                       BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_EIGEN_REFERENCE,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)))
// clang-format on
#undef SPCRAFT_DEFINE_EIGEN_REFERENCE
#undef SPCRAFT_DEFINE_EIGEN_REFERENCE_TYPES

}  // namespace spcraft
