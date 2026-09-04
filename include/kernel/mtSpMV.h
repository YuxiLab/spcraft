#pragma once

#include <stdexcept>
#include <type_traits>

#include "core/CsrMatrix.h"
#include "core/DenseVector.h"
#include "semiring/SemiRing.h"

namespace spcraft
{

/**
 * @brief Multiply a CSR matrix by a dense vector over a semiring using OpenMP.
 */
template <class SemiRing, class IT, class NT, class OT>
void spmv_openmp(const CsrMatrix<IT, NT, OT>& A, const DenseVector<IT, NT>& x,
                 DenseVector<IT, NT>& y)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "SpMV semiring value type must match the matrix value type");
  if (x.n != A.n || y.n != A.m) {
    throw std::invalid_argument("SpMV vector dimensions do not match the matrix");
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(A, x, y)
#endif
  for (IT row = 0; row < A.m; ++row) {
    NT sum = SemiRing::kAdditiveIdentity;
    for (OT position = A.row_ptr[row]; position < A.row_ptr[row + 1]; ++position) {
      sum = SemiRing::Add(sum, SemiRing::Multiply(A.val[position], x.val[A.col_id[position]]));
    }
    y.val[row] = sum;
  }
}

}  // namespace spcraft
