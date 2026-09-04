#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Reference OpenMP sparse-matrix sparse-matrix multiplication.
 *
 * Computes C = A * B over SemiRing with a row-wise map accumulator. Output
 * column indices are sorted within each CSR row. Structurally produced entries
 * are retained even when their final value equals the additive identity.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] CsrMatrix<IT, NT, OT> spgemm_openmp(const CsrMatrix<IT, NT, OT>& A,
                                                  const CsrMatrix<IT, NT, OT>& B)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "SpGEMM semiring value type must match the matrix value type");
  static_assert(std::is_integral_v<IT>, "SpGEMM index type must be integral");
  static_assert(std::is_integral_v<OT>, "SpGEMM offset type must be integral");

  if (A.n != B.m) {
    throw std::invalid_argument("SpGEMM matrix dimensions do not match");
  }
  if constexpr (std::is_signed_v<IT>) {
    if (A.m < 0 || A.n < 0 || B.m < 0 || B.n < 0) {
      throw std::invalid_argument("SpGEMM matrix dimensions must be non-negative");
    }
  }

  using Entry = std::pair<IT, NT>;
  std::vector<std::vector<Entry>> rows(static_cast<std::size_t>(A.m));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) default(none) shared(A, B, rows)
#endif
  for (IT row = 0; row < A.m; ++row) {
    std::map<IT, NT> accumulator;
    for (OT a_position = A.row_ptr[row]; a_position < A.row_ptr[row + 1]; ++a_position) {
      const IT inner = A.col_id[a_position];
      for (OT b_position = B.row_ptr[inner]; b_position < B.row_ptr[inner + 1]; ++b_position) {
        const IT column = B.col_id[b_position];
        auto [entry, inserted] = accumulator.try_emplace(column, SemiRing::kAdditiveIdentity);
        (void)inserted;
        entry->second = SemiRing::Add(
            entry->second, SemiRing::Multiply(A.val[a_position], B.val[b_position]));
      }
    }
    rows[static_cast<std::size_t>(row)].assign(accumulator.begin(), accumulator.end());
  }

  std::uintmax_t nonzeros = 0;
  constexpr std::uintmax_t kMaxOffset =
      static_cast<std::uintmax_t>(std::numeric_limits<OT>::max());
  for (const auto& row : rows) {
    if (row.size() > kMaxOffset - nonzeros) {
      throw std::overflow_error("SpGEMM output nnz exceeds the offset type");
    }
    nonzeros += row.size();
  }

  CsrMatrix<IT, NT, OT> result;
  result.Allocate(static_cast<OT>(nonzeros), A.m, B.n);
  result.row_ptr[0] = OT{0};
  for (IT row = 0; row < A.m; ++row) {
    result.row_ptr[row + 1] =
        result.row_ptr[row] + static_cast<OT>(rows[static_cast<std::size_t>(row)].size());
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(result, rows)
#endif
  for (IT row = 0; row < result.m; ++row) {
    OT position = result.row_ptr[row];
    for (const auto& [column, value] : rows[static_cast<std::size_t>(row)]) {
      result.col_id[position] = column;
      result.val[position] = value;
      ++position;
    }
  }

  return result;
}

}  // namespace spcraft
