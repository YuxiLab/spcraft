#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "utils/utils.h"

namespace spcraft
{

/**
 * @brief Reference omp spmv.
 */
SP_SR_MAT_TEMP
void OmpSpMV(const SPCSC& A, const SPDVEC& x, SPDVEC& y)
{
  if (x.n != A.n || y.n != A.m) {
    throw std::invalid_argument("SpMV vector dimensions do not match the matrix");
  }
  OMP_PARALLEL_FOR(schedule(dynamic))
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
