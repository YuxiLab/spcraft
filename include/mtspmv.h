#pragma once

#include "CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Multiply a CSR matrix by a dense vector using OpenMP.
 *
 * Computes `y = A * x`. Rows are independent, so each OpenMP loop iteration
 * owns one element of `y` and no synchronization is required. Empty rows
 * produce zero. When OpenMP is unavailable, the same loop remains a serial
 * fallback.
 *
 * The caller must provide at least `A.n` readable elements in `x` and at
 * least `A.m` writable elements in `y`. `x` and `y` must not overlap.
 *
 * @tparam IT Index type used by the matrix.
 * @tparam NT Numeric type used by the matrix and dense vectors.
 * @tparam OT Offset type used by the CSR row pointers.
 * @param[in] A CSR matrix.
 * @param[in] x Dense input vector.
 * @param[out] y Dense output vector.
 */
template <class IT, class NT, class OT>
void spmv_openmp(const CsrMatrix<IT, NT, OT>& A, const NT* x, NT* y)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(A, x, y)
#endif
  for (IT row = 0; row < A.m; ++row) {
    NT sum{};
    for (OT position = A.row_ptr[row]; position < A.row_ptr[row + 1]; ++position) {
      sum += A.val[position] * x[A.col_id[position]];
    }
    y[row] = sum;
  }
}

}  // namespace spcraft
