#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "core/CsrMatrix.h"
#include "utils/omp/omp_wrapper.h"
#include "utils/MatrixMacros.h"
namespace spcraft
{

/**
 * @brief Reference OpenMP SpMM.
 *
 */
SP_SR_MAT_TEMP
void spmm_openmp(const SPCSR& A, const NT* B, NT* C, std::size_t dense_columns)
{
  OMP_PARALLEL_FOR(schedule(static))
  for (IT row = 0; row < A.m; ++row) {
    const std::size_t output_row = static_cast<std::size_t>(row) * dense_columns;
    for (std::size_t column = 0; column < dense_columns; ++column) {
      NT sum = SemiRing::kAdditiveIdentity;
      for (OT position = A.row_ptr[row]; position < A.row_ptr[row + 1]; ++position) {
        const std::size_t input_row = static_cast<std::size_t>(A.col_id[position]) * dense_columns;
        sum = SemiRing::Add(sum, SemiRing::Multiply(A.val[position], B[input_row + column]));
      }
      C[output_row + column] = sum;
    }
  }
}

}  // namespace spcraft
