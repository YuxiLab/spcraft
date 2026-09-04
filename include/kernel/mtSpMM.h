#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "core/CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Reference OpenMP sparse-matrix dense-matrix multiplication.
 *
 * Computes C = A * B over SemiRing. A is an m-by-k CSR matrix, B is a
 * row-major k-by-dense_columns matrix, and C is a row-major
 * m-by-dense_columns matrix. B and C must not overlap.
 */
template <class SemiRing, class IT, class NT, class OT>
void spmm_openmp(const CsrMatrix<IT, NT, OT>& A, const NT* B, NT* C, std::size_t dense_columns)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "SpMM semiring value type must match the matrix value type");

  if (dense_columns != 0 && A.n != IT{0} && B == nullptr) {
    throw std::invalid_argument("SpMM input dense matrix must not be null");
  }
  if (dense_columns != 0 && A.m != IT{0} && C == nullptr) {
    throw std::invalid_argument("SpMM output dense matrix must not be null");
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(A, B, C, dense_columns)
#endif
  for (IT row = 0; row < A.m; ++row) {
    const std::size_t output_row = static_cast<std::size_t>(row) * dense_columns;
    for (std::size_t column = 0; column < dense_columns; ++column) {
      NT sum = SemiRing::kAdditiveIdentity;
      for (OT position = A.row_ptr[row]; position < A.row_ptr[row + 1]; ++position) {
        const std::size_t input_row =
            static_cast<std::size_t>(A.col_id[position]) * dense_columns;
        sum = SemiRing::Add(sum, SemiRing::Multiply(A.val[position], B[input_row + column]));
      }
      C[output_row + column] = sum;
    }
  }
}

}  // namespace spcraft
