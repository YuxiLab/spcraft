#pragma once

#include <cmath>
#include <cstdint>
#include "mtSpGEMM.h"
#include "utils/MatrixMacros.h"
#include "utils/omp/omp_numeric.h"
#include "utils/Print.h"
#include "utils/StdAliases.h"

namespace spcraft
{

SP_MAT_TEMP
VI HashSpGEMMCscSymbolicPhase(const SPCSC& A, const SPCSC& B, VL& flops_prefixsum)
{
  VI outnnz(B.n);  // get memory, but no initialization.
  OMP_PARALLEL_FOR()
  // looping the column of B
  for (IT j = 0; j < B.n; ++j) {
    OT b_cur_col_st_idx = B.col_ptr[j];      // get start index of B rowid at column j
    OT b_cur_col_ed_idx = B.col_ptr[j + 1];  // get start index of B rowid at column j+1
    OT capacity = 16;                        // minimum capacity
    OT cur_flops = flops_prefixsum[j + 1] - flops_prefixsum[j];
    while (capacity < cur_flops) {
      capacity <<= 1;  // hshtable capacity is pow of 2 and larger than flops
    }
    IT hashscale = 107;
    IT* hashkey = (IT*)malloc(sizeof(IT) * capacity);
    for (IT tt = 0; tt < capacity; ++tt) hashkey[tt] = -1;
    for (OT i = b_cur_col_st_idx; i < b_cur_col_ed_idx; ++i) {
      IT b_rowid = B.row_id[i];  // this is key for A's column
      for (OT k = A.col_ptr[b_rowid]; k < A.col_ptr[b_rowid + 1]; k++) {
        IT a_rowid = A.row_id[k];
        IT cur_key = ((size_t)a_rowid * hashscale) % capacity;  // init hash value.
        while (1) {
          if (hashkey[cur_key] == -1) {  // we hit an empty slot
            hashkey[cur_key] = a_rowid;
            break;
          } else if (hashkey[cur_key] != a_rowid) {
            cur_key = (cur_key + 1) % capacity;
          } else if (hashkey[cur_key] == a_rowid) {
            break;
          }
        }
      }
    }
    IT countnnz = 0;
    for (IT tt = 0; tt < capacity; ++tt) {
      if (hashkey[tt] != -1) countnnz++;
    }
    outnnz[j] = countnnz;
    free(hashkey);
  }
  return outnnz;
}

SP_MAT_TEMP
VL HashSpGEMMCscEstimateFlops(const SPCSC& A, const SPCSC& B)
{
  VL flops(B.n);  // give you memory, but no initialization.
  OMP_PARALLEL_FOR()
  for (IT j = 0; j < B.n; ++j) {  // looping the column of B
    int64_t current_column_flops = 0;
    OT brow_start = B.col_ptr[j];     // get start index of B rowid at column j
    OT brow_ends = B.col_ptr[j + 1];  // get start index of B rowid at column j+1
    for (OT i = brow_start; i < brow_ends; ++i) {
      // use B rowid as key, to find the nnz of corresponding A column and sum
      current_column_flops += A.col_ptr[B.row_id[i] + 1] - A.col_ptr[B.row_id[i]];
    }
    flops[j] = current_column_flops;
  }
  return flops;
}

/**
 * @brief Fill preallocated COO column slices using the symbolic NNZ offsets.
 * Entries are sorted by row within each column; structural zeros are retained.
 */
SP_SR_MAT_TEMP
void HashSPGEMMCscNumericPhase(const SPCSC& A, const SPCSC& B, const VL& offsets, SPCOO& result)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");
  struct HashKeyEntry {
    IT key;
    NT value;
  };
  constexpr IT kEmpty = static_cast<IT>(-1);
  constexpr OT kMinCapacity = 16;
  constexpr IT kHashScale = 107;
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT j = 0; j < B.n; ++j) {
    const auto count = offsets[j + 1] - offsets[j];
    if (count == 0) continue;
    OT capacity = kMinCapacity;
    while (capacity < count) capacity <<= 1;
    std::vector<HashKeyEntry> table(capacity, {kEmpty, SemiRing::kAdditiveIdentity});
    for (OT i = B.col_ptr[j]; i < B.col_ptr[j + 1]; ++i) {
      const IT column = B.row_id[i];
      for (OT k = A.col_ptr[column]; k < A.col_ptr[column + 1]; ++k) {
        const IT row = A.row_id[k];
        IT slot = (static_cast<std::size_t>(row) * kHashScale) % capacity;
        while (table[slot].key != kEmpty && table[slot].key != row) {
          slot = (slot + 1) % capacity;
        }
        table[slot].key = row;
        table[slot].value =
            SemiRing::Add(table[slot].value, SemiRing::Multiply(A.val[k], B.val[i]));
      }
    }
    std::size_t occupied = 0;
    for (std::size_t slot = 0; slot < capacity; ++slot) {
      if (table[slot].key != kEmpty) table[occupied++] = table[slot];
    }
    std::sort(table.begin(), table.begin() + occupied,
              [](const HashKeyEntry& a, const HashKeyEntry& b) { return a.key < b.key; });
    OT dest = static_cast<OT>(offsets[j]);
    for (std::size_t slot = 0; slot < occupied; ++slot) {
      result.entries[dest++] = {table[slot].key, j, table[slot].value};
    }
  }
}

SP_SR_MAT_TEMP
SPND SPCOO OmpHashSpGEMM(const SPCSC& A, const SPCSC& B)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");
  if (A.n != B.m) throw std::invalid_argument("SpGEMM matrix dimensions do not match");
  if (A.m < 0 || A.n < 0 || B.m < 0 || B.n < 0)
    throw std::invalid_argument("SpGEMM dimensions must be nonnegative");
  SPCOO result;
  if (!A.nnz || !B.nnz) {
    result.Allocate(0, A.m, B.n);
    return result;
  }
  const int threads = OMP_GET_MAX_THREADS();
  // constexpr std::size_t kPrintCount = 5;
  VL flop = HashSpGEMMCscEstimateFlops(A, B);
  // PrintVector(flop, kPrintCount, "flops", std::cerr);
  // Keep this work prefix sum to match the CombBLAS baseline.
  VL flop_prefixsum = OmpPrefixSum(flop, threads);
  // PrintVector(flop_prefixsum, kPrintCount, "flops prefixsum", std::cerr);
  // 3. Symbolic phase: count unique rows and assign each column a COO slice.
  auto counts = HashSpGEMMCscSymbolicPhase(A, B, flop_prefixsum);
  // PrintVector(counts, kPrintCount, "nnz per column", std::cerr);
  // Per-column counts fit int32_t, but their total may need 64-bit offsets.
  VL offsets = OmpPrefixSum(VL(counts.begin(), counts.end()), threads);
  if (static_cast<std::uintmax_t>(offsets.back()) >
      static_cast<std::uintmax_t>(std::numeric_limits<OT>::max())) {
    throw std::overflow_error("SpGEMM output nnz exceeds offset type limit");
  }
  // PrintVector(offsets, kPrintCount, "nnz offsets", std::cerr);
  result.Allocate(static_cast<OT>(offsets.back()), A.m, B.n);
  HashSPGEMMCscNumericPhase<SemiRing>(A, B, offsets, result);
  return result;
}

}  // namespace spcraft
