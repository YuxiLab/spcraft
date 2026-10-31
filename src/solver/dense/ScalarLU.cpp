#include "solver/dense/Lu.h"

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include "core/DenseVector.h"
#include "solver/Lu.h"

namespace spcraft::solver::dense
{

template <class IT, class NT, class OT>
void ForwardSolve(const DenseMatrix<IT, NT, OT>& lu, const DenseVector<IT, NT>& rhs, DenseVector<IT, NT>& y)
{
  // Solve L*y=rhs; L has an implicit unit diagonal.
  // Assume valid shapes and disjoint output storage.
  // 1. Solve row k using the already computed y[i] for i<k.
  //    Start with rhs[k] and subtract lu[k,i]*y[i] for each earlier row i.
  // 2. Store y[k] directly; lu[k,k] is U's diagonal, not L's.
  for (IT k = 0; k < y.n; ++k) {
    NT tmp = rhs[k];
    for (IT i = 0; i < k; ++i) {
      tmp -= lu[k, i] * y[i];
    }
    y[k] = tmp;
  }
}

template <class IT, class NT, class OT>
void BackwardSolve(const DenseMatrix<IT, NT, OT>& lu, const DenseVector<IT, NT>& rhs, DenseVector<IT, NT>& x)
{
  // Solve U*x=rhs using the upper triangle of the packed factors.
  // Assume valid shapes and disjoint output storage.
  // 1. Visit row k from last to first, using the already computed x[i] for i>k.
  //    Start with rhs[k] and subtract lu[k,i]*x[i] for each later row i.
  // 2. Divide the residual by lu[k,k] to store x[k].
  //    Reject a zero or nonfinite diagonal with runtime_error before division.
  for (IT k = rhs.n - 1; k >= 0; --k) {
    NT tmp = rhs[k];
    for (IT i = k + 1; i < rhs.n; ++i) {
      tmp -= lu[k, i] * x[i];
    }
    x[k] = tmp / lu[k, k];
  }
}

template <class IT, class NT, class OT>
Factors<IT, NT, OT> RightLookingFactorize(const DenseMatrix<IT, NT, OT>& A, const LuOptions<NT>& options)
{
  // Factor A=L*U with right-looking scalar updates to the trailing matrix.
  // Assume valid square A, finite coefficients, and a finite nonnegative tolerance.
  // 1. Clone A into the owning result.lu so the original input stays unchanged.
  // 2. Process pivot columns k in order using the updated diagonal lu[k,k].
  //    Return ZeroPivot at k before division if it is nonfinite or within tolerance.
  // 3. Store lu[i,k]/lu[k,k] as the multiplier for each row i>k.
  //    Subtract lu[i,k]*lu[k,j] from each trailing entry with i,j>k.
  // 4. Mark Success with pivot_column=-1 after all columns finish.
  //    Keep L's diagonal implicit; the stored diagonal belongs to U.
  Factors<IT, NT, OT> factors;
  factors.lu = A.Clone();
  auto& lu = factors.lu;
  for (IT k = 0; k < A.n; ++k) {
    const NT pivot = lu[k, k];
    if (!std::isfinite(pivot) || std::abs(pivot) <= options.pivot_tolerance) {
      factors.status.code = LuCode::ZeroPivot;
      factors.status.pivot_column = k;
      return factors;
    }
    for (IT i = k + 1; i < A.m; ++i) {
      lu[i, k] /= lu[k, k];
    }
    for (IT i = k + 1; i < A.m; ++i) {
      for (IT j = k + 1; j < A.n; ++j) {
        lu[i, j] -= lu[i, k] * lu[k, j];
      }
    }
  }
  factors.status.code = LuCode::Success;
  return factors;
}

template <class IT, class NT, class OT>
Factors<IT, NT, OT> LeftLookingFactorize(const DenseMatrix<IT, NT, OT>& A, const LuOptions<NT>& options)
{
  // Factor A=L*U column by column, gathering updates from earlier L columns.
  // Assume valid square A, finite coefficients, and a finite nonnegative tolerance.
  // 1. Clone A into owning factors and visit the current column k in order.
  // 2. For each earlier column j<k, read the updated lu[j,k] as U(j,k).
  //    Subtract lu[i,j]*lu[j,k] from lu[i,k] for rows i>j before advancing j.
  // 3. Check the updated pivot lu[k,k]; return ZeroPivot before division on failure.
  //    Reject a nonfinite pivot or abs(pivot)<=options.pivot_tolerance.
  // 4. Divide entries lu[i,k] by the pivot for i>k to store L's multipliers.
  //    Update only column k; future columns gather updates when reached.
  // 5. Mark Success with pivot_column=-1 after all columns finish.
  // Comment out the placeholder exception after implementing this function.
  Factors<IT, NT, OT> factors;
  factors.lu = A.Clone();
  auto& lu = factors.lu;

  throw std::logic_error("LeftLookingFactorize: implement column-oriented delayed updates");
}

template <class IT, class NT, class OT>
Factors<IT, NT, OT> UpperLookingFactorize(const DenseMatrix<IT, NT, OT>& A, const LuOptions<NT>& options)
{
  // Factor A=L*U row by row using completed U rows; this is up-looking LU.
  // Assume valid square A, finite coefficients, and a finite nonnegative tolerance.
  // 1. Clone A into owning factors and visit the current row k in order.
  // 2. For each earlier row j<k, divide the updated lu[k,j] by lu[j,j].
  //    This stores L(k,j); previous rows have already checked their pivots.
  // 3. Subtract lu[k,j]*lu[j,i] from lu[k,i] for columns i>j before advancing j.
  //    Update only row k; later rows gather updates when reached.
  // 4. Check the completed row's pivot lu[k,k] and return ZeroPivot on failure.
  //    Reject a nonfinite pivot or abs(pivot)<=options.pivot_tolerance.
  // 5. Mark Success with pivot_column=-1 after all rows finish.
  // Comment out the placeholder exception after implementing this function.
  throw std::logic_error("UpperLookingFactorize: implement row-oriented up-looking updates");
}

template <class IT, class NT, class OT>
void Solve(const Factors<IT, NT, OT>& factors, const DenseVector<IT, NT>& b, DenseVector<IT, NT>& x)
{
  // Solve A*x=b using existing packed factors, without repeating elimination.
  // Assume successful factors, matching vector sizes, and disjoint output storage.
  // 1. Allocate a private y and call ForwardSolve to compute L*y=b.
  // 2. Call BackwardSolve to compute U*x=y.
  //    Preserve b, the factor matrix, and its status for another right-hand side.
  DenseVector<IT, NT> y(x.n);
  ForwardSolve(factors.lu, b, y);
  BackwardSolve(factors.lu, y, x);
}
// dont modifiy it
#define SPCRAFT_DEFINE_DENSE_LU_TYPES(IT, NT, OT)                                                           \
  template void ForwardSolve(const DenseMatrix<IT, NT, OT>&, const DenseVector<IT, NT>&,                    \
                             DenseVector<IT, NT>&);                                                         \
  template void BackwardSolve(const DenseMatrix<IT, NT, OT>&, const DenseVector<IT, NT>&,                   \
                              DenseVector<IT, NT>&);                                                        \
  template Factors<IT, NT, OT> RightLookingFactorize(const DenseMatrix<IT, NT, OT>&, const LuOptions<NT>&); \
  template Factors<IT, NT, OT> LeftLookingFactorize(const DenseMatrix<IT, NT, OT>&, const LuOptions<NT>&);  \
  template Factors<IT, NT, OT> UpperLookingFactorize(const DenseMatrix<IT, NT, OT>&, const LuOptions<NT>&); \
  template void Solve(const Factors<IT, NT, OT>&, const DenseVector<IT, NT>&, DenseVector<IT, NT>&);

#define SPCRAFT_DEFINE_DENSE_LU(r, product)                                          \
  SPCRAFT_DEFINE_DENSE_LU_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                BOOST_PP_SEQ_ELEM(1, product),                       \
                                BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_DENSE_LU,
                              (((int32_t)(int32_t))
                               ((int32_t)(int64_t))
                               ((int64_t)(int64_t)))
                              ((double)))
// clang-format on
#undef SPCRAFT_DEFINE_DENSE_LU
#undef SPCRAFT_DEFINE_DENSE_LU_TYPES

}  // namespace spcraft::solver::dense
