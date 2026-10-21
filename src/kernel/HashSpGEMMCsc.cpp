#include <cmath>
#include <cstdint>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "kernel/mtSpGEMM.h"
#include "semiring/SemiRing.h"
#include "utils/MatrixMacros.h"
#include "utils/omp/omp_numeric.h"
#include "utils/Print.h"
#include "utils/StdAliases.h"

namespace spcraft
{
namespace
{


// @brief 
// Note: this is symbolic phase of hash-based spgemm. It determines the EXACT nnz of a single column in 
// output C matrix. 
// for per column variable we use IT, for the prefx sum variables we use OT.
template <class IT, class NT, class OT>
V<IT> HashSpGEMMCscSymbolicPhase(const SPCSC& A, const SPCSC& B, const V<OT>& flops_prefixsum)
{
  constexpr IT kMinCapacity = 16;
  constexpr IT kHashScale = 107;
  constexpr IT kEmpty = -1;
  V<IT> counts(B.n);
  // Each column owns its scratch; irregular expansions are scheduled dynamically.
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT column = 0; column < B.n; ++column) {
    const OT begin = B.col_ptr[column];
    const OT end = B.col_ptr[column + 1];
    const IT flops = flops_prefixsum[column + 1] - flops_prefixsum[column];
    IT htsize = kMinCapacity;
    while (htsize < flops) htsize <<= 1;
    std::vector<IT> keys(htsize, kEmpty);
    for (OT p = begin; p < end; ++p) {
      const IT a_column = B.row_id[p];
      for (OT q = A.col_ptr[a_column]; q < A.col_ptr[a_column + 1]; ++q) {
        const IT row = A.row_id[q];
        IT slot = (row * kHashScale) % htsize;
        while (keys[slot] != kEmpty && keys[slot] != row) slot = (slot + 1) % htsize;
        keys[slot] = row;
      }
    }
    int32_t count = 0;
    for (IT slot = 0; slot < htsize; ++slot) {
      if (keys[slot] != kEmpty) ++count;
    }
    counts[column] = count;
  }
  return counts;
}

template <class IT, class NT, class OT>
V<OT> HashSpGEMMCscEstimateFlops(const SPCSC& A, const SPCSC& B)
{
  V<OT> flops(B.n);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT j = 0; j < B.n; ++j) {
    int64_t current_column_flops = 0;
    const OT brow_start = B.col_ptr[j];
    const OT brow_ends = B.col_ptr[j + 1];
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
template <class SemiRing, class IT, class NT, class OT>
void HashSPGEMMCscNumericPhase(const SPCSC& A, const SPCSC& B, const V<OT>& offsets, SPCOO& result)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");
  struct HashKeyEntry {
    /***************************************************************************
     *                              Data Members                               *
     **************************************************************************/
    IT key = static_cast<IT>(-1);
    NT value = SemiRing::kAdditiveIdentity;
  };
  constexpr IT kEmpty = static_cast<IT>(-1);
  constexpr OT kMinCapacity = 16;
  constexpr IT kHashScale = 107;
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT j = 0; j < B.n; ++j) {
    const int64_t count = offsets[j + 1] - offsets[j];
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
        table[slot].value = SemiRing::Add(table[slot].value, SemiRing::Multiply(A.val[k], B.val[i]));
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

}  // namespace

template <class SemiRing, class IT, class NT, class OT>
SPND SPCOO OmpHashSpGEMM(const SPCSC& A, const SPCSC& B)
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

  // 1. estimate flops
  // @profiler-region hash_estimate_flops start
  V<OT> flop = HashSpGEMMCscEstimateFlops(A, B);
  // @profiler-region hash_estimate_flops end

  // 2. Keep this work prefix sum to match the CombBLAS baseline.
  // @profiler-region hash_flops_prefix start
  V<OT> flop_prefixsum = OmpPrefixSum(flop, threads);
  // @profiler-region hash_flops_prefix end

  // 3. Symbolic phase: count unique rows and assign each column a COO slice.
  // @profiler-region hash_symbolic start
  V<IT> counts = HashSpGEMMCscSymbolicPhase(A, B, flop_prefixsum);
  // @profiler-region hash_symbolic end

  // 4. prefix sum of nnz per column. Per-column counts fit int32_t, but their total may need 64-bit offsets.
  // @profiler-region hash_output_offsets start
  const V<IT> wide_counts(counts.begin(), counts.end());
  const V<OT> offsets = OmpPrefixSum<IT,OT>(wide_counts, threads);
  if (static_cast<std::uintmax_t>(offsets.back()) >
      static_cast<std::uintmax_t>(std::numeric_limits<OT>::max())) {
    throw std::overflow_error("SpGEMM output nnz exceeds offset type limit");
  }
  // @profiler-region hash_output_offsets end

  // @profiler-region hash_allocate start
  result.Allocate(static_cast<OT>(offsets.back()), A.m, B.n);
  // @profiler-region hash_allocate end

  // @profiler-region hash_numeric start
  HashSPGEMMCscNumericPhase<SemiRing>(A, B, offsets, result);
  // @profiler-region hash_numeric end

  return result;
}

#define SPCRAFT_DEFINE_HASH_SPGEMM_CSC_TYPES(IT, NT, OT)                                                    \
  template CooMatrix<IT, NT, OT> OmpHashSpGEMM<PlusTimesRing<NT>, IT, NT, OT>(const CscMatrix<IT, NT, OT>&, \
                                                                              const CscMatrix<IT, NT, OT>&);

#define SPCRAFT_DEFINE_HASH_SPGEMM_CSC(r, product)                                          \
  SPCRAFT_DEFINE_HASH_SPGEMM_CSC_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                       BOOST_PP_SEQ_ELEM(1, product),                       \
                                       BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// Pair the storage types so 64-bit indices never select 32-bit offsets.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_HASH_SPGEMM_CSC,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)))
// clang-format on
#undef SPCRAFT_DEFINE_HASH_SPGEMM_CSC
#undef SPCRAFT_DEFINE_HASH_SPGEMM_CSC_TYPES

}  // namespace spcraft
