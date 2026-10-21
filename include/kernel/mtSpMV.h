#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "utils/utils.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Compute y = A * x over a semiring using OpenMP CSR row traversal.
 */
template <class SemiRing, class IT, class NT, class OT>
void OmpSpMV(const SPCSR& A, const SPDVEC& x, SPDVEC& y)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");

  if (x.n != A.n || y.n != A.m) {
    throw std::invalid_argument("SpMV vector dimensions do not match the matrix");
  }
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT row = 0; row < A.m; ++row) {
    NT sum = SemiRing::kAdditiveIdentity;
    for (OT a_col = A.row_ptr[row]; a_col < A.row_ptr[row + 1]; ++a_col) {
      const NT aval = A.val[a_col];
      const NT xval = x.val[A.col_id[a_col]];
      sum = SemiRing::Add(sum, SemiRing::Multiply(aval, xval));
    }
    y.val[row] = sum;
  }
}

}  // namespace spcraft
