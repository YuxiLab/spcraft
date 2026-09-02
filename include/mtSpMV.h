#pragma once

#include "CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Multiply a CSR matrix by a dense vector using OpenMP.
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
