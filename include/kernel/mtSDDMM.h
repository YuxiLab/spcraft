#pragma once

#include "utils/omp/omp_wrapper.h"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "core/CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Reference OpenMP sampled dense-dense matrix multiplication.
 *
 * For every nonzero (row, column) in samples, computes the semiring dot
 * product of left[row, :] and right[column, :] and stores it in samples.val.
 * left and right are row-major matrices with feature_count columns.
 */
template <class SemiRing, class IT, class NT, class OT>
void sddmm_openmp(CsrMatrix<IT, NT, OT>& samples, const NT* left, const NT* right,
                  std::size_t feature_count)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "SDDMM semiring value type must match the matrix value type");

  if (feature_count != 0 && samples.m != IT{0} && left == nullptr) {
    throw std::invalid_argument("SDDMM left dense matrix must not be null");
  }
  if (feature_count != 0 && samples.n != IT{0} && right == nullptr) {
    throw std::invalid_argument("SDDMM right dense matrix must not be null");
  }
  if (samples.nnz != OT{0} && samples.val == nullptr) {
    throw std::invalid_argument("SDDMM output values must not be null");
  }

  OMP_PARALLEL_FOR(schedule(static) default(none) shared(samples, left, right, feature_count))
  for (IT row = 0; row < samples.m; ++row) {
    const std::size_t left_row = static_cast<std::size_t>(row) * feature_count;
    for (OT position = samples.row_ptr[row]; position < samples.row_ptr[row + 1]; ++position) {
      const std::size_t right_row =
          static_cast<std::size_t>(samples.col_id[position]) * feature_count;
      NT sum = SemiRing::kAdditiveIdentity;
      for (std::size_t feature = 0; feature < feature_count; ++feature) {
        sum = SemiRing::Add(
            sum, SemiRing::Multiply(left[left_row + feature], right[right_row + feature]));
      }
      samples.val[position] = sum;
    }
  }
}

}  // namespace spcraft
