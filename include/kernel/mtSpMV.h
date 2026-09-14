#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "SpCraft.h"

namespace spcraft
{

/**
 * @brief Reference omp spmv.
 */
template <class SemiRing, class IT, class NT, class OT>
void OmpSpMV(const CsrMatrix<IT, NT, OT>& A, const DenseVector<IT, NT>& x, DenseVector<IT, NT>& y)
{
  // clang-format off
  if (x.n != A.n || y.n != A.m) { throw std::invalid_argument("SpMV vector dimensions do not match the matrix"); }
  // clang-format on
  OMP_PARALLEL_FOR()
  // loop each row in A matrix.
  for (IT row = 0; row < A.m; ++row) {
    // sum is variable store the output of the dot product.
    NT sum = SemiRing::kAdditiveIdentity;
    // get the column id for each nnz in that row.
    for (OT a_col = A.row_ptr[row]; a_col < A.row_ptr[row + 1]; ++a_col) {
      const NT aval = A.val[a_col];            // the value of current nnz element
      const NT xval = x.val[A.col_id[a_col]];  // the value of corresponding x value
      // add into the sum variable.
      sum = SemiRing::Add(sum, SemiRing::Multiply(aval, xval));
    }
    // finish dot product, store the result to output y vector.
    y.val[row] = sum;
  }
}

}  // namespace spcraft
