#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include "kernel/mtSpGEMM.h"

namespace spcraft
{
namespace detail
{

/**
 * @brief CombBLAS-style chunk lookup with O(A.nzc) storage, built for nonempty A.
 */
SP_MAT_TEMP
class DcscColumnLookup
{
 public:
  explicit DcscColumnLookup(const SPDCSC& A) : matrix(A)
  {
    // Integer ceiling division avoids losing column IDs through float rounding.
    const int64_t extent = static_cast<int64_t>(A.n) + 1;
    chunk_size = (extent - 1) / A.nzc + 1;
    const int64_t chunks = (extent - 1) / chunk_size + 1;
    aux.resize(CheckedElementCount<IT>(chunks + 1));
    int64_t next_chunk = 1;
    for (IT col = 0; col < A.nzc; ++col) {
      const int64_t chunk = A.col_id[col] / chunk_size;
      while (next_chunk <= chunk) aux[next_chunk++] = col;
    }
    while (next_chunk <= chunks) aux[next_chunk++] = A.nzc;
  }

  std::vector<std::pair<OT, OT>> FillColInds(const SPDCSC& B, IT col) const
  {
    const OT begin = B.col_ptr[col], end = B.col_ptr[col + 1];
    const auto count = static_cast<std::size_t>(end - begin);
    std::vector<std::pair<OT, OT>> ranges(count);
    if (count == 0) return ranges;

    // CombBLAS switches to scanning when many A columns are requested. Our
    // containers also permit unsorted rows, which must use independent lookups.
    constexpr std::size_t kScanThreshold = 4;
    const bool scan = static_cast<std::size_t>(matrix.nzc) / count < kScanThreshold &&
                      std::is_sorted(B.row_id + begin, B.row_id + end);
    if (scan) {
      IT pos = 0;
      for (OT p = begin; p < end; ++p) {
        const IT column = B.row_id[p];
        while (pos < matrix.nzc && matrix.col_id[pos] < column) ++pos;
        if (pos < matrix.nzc && matrix.col_id[pos] == column) {
          ranges[p - begin] = {matrix.col_ptr[pos], matrix.col_ptr[pos + 1]};
        }
        // Keep pos at a match so repeated B row IDs reuse the same A column.
      }
    } else {
      for (OT p = begin; p < end; ++p) {
        const IT column = B.row_id[p];
        const auto chunk = column / chunk_size;
        for (IT pos = aux[chunk]; pos < aux[chunk + 1]; ++pos) {
          if (matrix.col_id[pos] == column) {
            ranges[p - begin] = {matrix.col_ptr[pos], matrix.col_ptr[pos + 1]};
            break;
          }
          if (matrix.col_id[pos] > column) break;
        }
      }
    }
    return ranges;
  }

 private:
  const SPDCSC& matrix;
  VIT aux;
  int64_t chunk_size;
};

}  // namespace detail

SP_MAT_TEMP
VL HashSpGEMM_Csc_EstimateFlops(const SPDCSC& A, const SPDCSC& B,
                                const detail::DcscColumnLookup<IT, NT, OT>& lookup)
{
  VL flops(B.nzc, 0);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT col = 0; col < B.nzc; ++col) {
    const auto ranges = lookup.FillColInds(B, col);
    int64_t count = 0;
    for (const auto& [begin, end] : ranges) count += end - begin;
    flops[col] = count;
  }
  return flops;
}

SP_MAT_TEMP
VL HashpSpGEMM_Csc_SymbolicPhase(const SPDCSC& A, const SPDCSC& B, const VL& flop_prefixsum,
                                 const detail::DcscColumnLookup<IT, NT, OT>& lookup)
{
  VL counts(B.nzc, 0);
  constexpr std::size_t kMinCapacity = 16;
  constexpr std::size_t kHashScale = 107;
  constexpr IT kEmpty = static_cast<IT>(-1);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT col = 0; col < B.nzc; ++col) {
    const auto flops = flop_prefixsum[col + 1] - flop_prefixsum[col];
    if (flops == 0) continue;
    std::size_t capacity = kMinCapacity;
    while (capacity < static_cast<std::size_t>(flops)) capacity <<= 1;
    const auto mask = capacity - 1;
    std::vector<IT> keys(capacity, kEmpty);
    const auto ranges = lookup.FillColInds(B, col);
    int64_t count = 0;
    for (const auto& [begin, end] : ranges) {
      for (OT p = begin; p < end; ++p) {
        const IT row = A.row_id[p];
        auto slot = (static_cast<std::size_t>(row) * kHashScale) & mask;
        while (keys[slot] != kEmpty && keys[slot] != row) slot = (slot + 1) & mask;
        if (keys[slot] == kEmpty) {
          keys[slot] = row;
          ++count;
        }
      }
    }
    counts[col] = count;
  }
  return counts;
}

/**
 * @brief Accumulate and sort each stored B column into its reserved COO slice.
 */
SP_SR_MAT_TEMP
void HashSpGEMM_Dcsc_NumericPhase(const SPDCSC& A, const SPDCSC& B, const VL& offsets,
                                  const detail::DcscColumnLookup<IT, NT, OT>& lookup, SPCOO& result)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");
  struct HashKeyEntry {
    IT key;
    NT value;
  };
  constexpr std::size_t kMinCapacity = 16;
  constexpr std::size_t kHashScale = 107;
  constexpr IT kEmpty = static_cast<IT>(-1);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT col = 0; col < B.nzc; ++col) {
    const auto count = offsets[col + 1] - offsets[col];
    if (count == 0) continue;
    std::size_t capacity = kMinCapacity;
    while (capacity < static_cast<std::size_t>(count)) capacity <<= 1;
    const auto mask = capacity - 1;
    std::vector<HashKeyEntry> table(capacity, {kEmpty, SemiRing::kAdditiveIdentity});
    const auto ranges = lookup.FillColInds(B, col);
    const OT begin = B.col_ptr[col], end = B.col_ptr[col + 1];
    for (OT p = begin; p < end; ++p) {
      const auto& [a_begin, a_end] = ranges[p - begin];
      for (OT q = a_begin; q < a_end; ++q) {
        const IT row = A.row_id[q];
        auto slot = (static_cast<std::size_t>(row) * kHashScale) & mask;
        while (table[slot].key != kEmpty && table[slot].key != row) slot = (slot + 1) & mask;
        table[slot].key = row;
        table[slot].value =
            SemiRing::Add(SemiRing::Multiply(A.val[q], B.val[p]), table[slot].value);
      }
    }
    std::size_t occupied = 0;
    for (std::size_t slot = 0; slot < capacity; ++slot) {
      if (table[slot].key != kEmpty) table[occupied++] = table[slot];
    }
    std::sort(table.begin(), table.begin() + occupied,
              [](const HashKeyEntry& a, const HashKeyEntry& b) { return a.key < b.key; });
    OT dest = static_cast<OT>(offsets[col]);
    for (std::size_t slot = 0; slot < occupied; ++slot) {
      result.entries[dest++] = {table[slot].key, B.col_id[col], table[slot].value};
    }
  }
}

/**
 * @brief CombBLAS-style DCSC hash SpGEMM with sorted COO output.
 * Builds one O(A.nzc) column lookup shared by work estimation, symbolic counting,
 * and numeric accumulation. Rows may be unsorted or repeated; structural zeros
 * are retained. Work/output prefixes use int64_t, with OT-checked output storage.
 */
SP_SR_MAT_TEMP
SPND SPCOO OmpHashSpGEMM(const SPDCSC& A, const SPDCSC& B)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");
  if (A.n != B.m) throw std::invalid_argument("SpGEMM matrix dimensions do not match");
  if constexpr (std::is_signed_v<IT>) {
    if (A.m < 0 || A.n < 0 || B.m < 0 || B.n < 0)
      throw std::invalid_argument("SpGEMM dimensions must be nonnegative");
  }
  SPCOO result;
  if (!A.nnz || !B.nnz) {
    result.Allocate(0, A.m, B.n);
    return result;
  }
  const int threads = OMP_GET_MAX_THREADS();
  const detail::DcscColumnLookup lookup(A);
  const auto flops = HashSpGEMM_Csc_EstimateFlops(A, B, lookup);
  const auto flop_prefixsum = OmpPrefixSum(flops, threads);
  const auto counts = HashpSpGEMM_Csc_SymbolicPhase(A, B, flop_prefixsum, lookup);

  const auto offsets = OmpPrefixSum(counts, threads);
  if (static_cast<std::uintmax_t>(offsets.back()) >
      static_cast<std::uintmax_t>(std::numeric_limits<OT>::max())) {
    throw std::overflow_error("SpGEMM output nnz exceeds offset type limit");
  }
  result.Allocate(static_cast<OT>(offsets.back()), A.m, B.n);
  HashSpGEMM_Dcsc_NumericPhase<SemiRing>(A, B, offsets, lookup, result);
  return result;
}

}  // namespace spcraft
