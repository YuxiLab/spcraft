#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "core/AllocationGuard.h"
#include "kernel/mtSpGEMM.h"
#include "semiring/SemiRing.h"
#include "utils/MatrixMacros.h"
#include "utils/StdAliases.h"

namespace spcraft
{
namespace
{

/**
 * @brief CombBLAS-style chunk lookup with O(A.nzc) storage, built for nonempty A.
 */
template <class IT, class NT, class OT>
class DcscColumnLookup
{
 public:
  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  explicit DcscColumnLookup(const SPDCSC& A) : A(A)
  {
    // Integer ceiling division avoids losing column IDs through float rounding.
    const IT extent = A.n + 1;
    chunk_size = (extent - 1) / A.nzc + 1;
    const IT chunks = (extent - 1) / chunk_size + 1;
    aux.resize(chunks + 1);
    IT next_chunk = 1;
    for (IT col = 0; col < A.nzc; ++col) {
      const IT chunk = A.col_id[col] / chunk_size;
      while (next_chunk <= chunk) aux[next_chunk++] = col;
    }
    while (next_chunk <= chunks) aux[next_chunk++] = A.nzc;
  }

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  VPair<OT, OT> FillColInds(const SPDCSC& B, IT col) const
  {
    const OT begin = B.col_ptr[col];
    const OT end = B.col_ptr[col + 1];
    const OT count = end - begin;
    VPair<OT, OT> ranges(count);
    if (count == 0) return ranges;

    constexpr IT kScanThreshold = 4;
    const bool scan = (OT)A.nzc / count < kScanThreshold;
    // if count is large -> a linear scan is enough
    if (scan) {
      IT pos = 0;
      for (OT p = begin; p < end; ++p) {
        const IT column = B.row_id[p];
        while (pos < A.nzc && A.col_id[pos] < column) ++pos;
        if (pos < A.nzc && A.col_id[pos] == column) {
          ranges[p - begin] = {A.col_ptr[pos], A.col_ptr[pos + 1]};
        }
      }
    } else {
      // if count is small -> use chunk data structure
      for (OT p = begin; p < end; ++p) {
        const IT column = B.row_id[p];
        const int64_t chunk = column / chunk_size;
        for (IT pos = aux[chunk]; pos < aux[chunk + 1]; ++pos) {
          if (A.col_id[pos] == column) {
            ranges[p - begin] = {A.col_ptr[pos], A.col_ptr[pos + 1]};
            break;
          }
          if (A.col_id[pos] > column) break;
        }
      }
    }
    return ranges;
  }

 private:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  const SPDCSC& A;
  V<IT> aux{};
  IT chunk_size = 0;
};

template <class IT, class NT, class OT>
V<OT> EstimateDcscFlops(const SPDCSC& A, const SPDCSC& B, const DcscColumnLookup<IT, NT, OT>& lookup)
{
  V<OT> flops(B.nzc, 0);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT col = 0; col < B.nzc; ++col) {
    const VPair<OT, OT> ranges = lookup.FillColInds(B, col);
    int64_t count = 0;
    for (const std::pair<OT, OT>& range : ranges) count += range.second - range.first;
    flops[col] = count;
  }
  return flops;
}

template <class IT, class NT, class OT>
V<IT> HashSpGEMMDcscSymbolicPhase(const SPDCSC& A, const SPDCSC& B, const V<OT>& flop_prefixsum,
                                  const DcscColumnLookup<IT, NT, OT>& lookup)
{
  // yuxi: in the symbolic phase, only flop prefix sum is OT. others are all single 
  // column conuter, so they should not exceed the IT range. 
  V<IT> counts(B.nzc, 0);
  constexpr IT kMinCapacity = 16;
  constexpr IT kHashScale = 107;
  constexpr IT kEmpty = -1;
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT col = 0; col < B.nzc; ++col) {
    const IT flops = flop_prefixsum[col + 1] - flop_prefixsum[col];
    if (flops == 0) continue;
    IT capacity = kMinCapacity;
    while (capacity < flops) capacity <<= 1;
    const IT mask = capacity - 1;
    V<IT> keys(capacity, kEmpty);
    const VPair<OT, OT> ranges = lookup.FillColInds(B, col);
    IT count = 0;
    for (const std::pair<OT, OT>& range : ranges) {
      for (OT p = range.first; p < range.second; ++p) {
        const IT row = A.row_id[p];
        IT slot = (row * kHashScale) & mask;
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
void HashSpGEMMDcscNumericPhase(const SPDCSC& A, const SPDCSC& B, const V<OT>& offsets,
                                const DcscColumnLookup<IT, NT, OT>& lookup, SPCOO& result)
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
  constexpr IT kMinCapacity = 16;
  constexpr IT kHashScale = 107;
  constexpr IT kEmpty = static_cast<IT>(-1);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT col = 0; col < B.nzc; ++col) {
    const IT count = offsets[col + 1] - offsets[col];
    if (count == 0) continue;
    IT capacity = kMinCapacity;
    while (capacity < count) capacity <<= 1;
    const IT mask = capacity - 1;
    std::vector<HashKeyEntry> table(capacity, {kEmpty, SemiRing::kAdditiveIdentity});
    const VPair<OT, OT> ranges = lookup.FillColInds(B, col);
    const OT begin = B.col_ptr[col];
    const OT end = B.col_ptr[col + 1];
    for (OT p = begin; p < end; ++p) {
      const std::pair<OT, OT>& range = ranges[p - begin];
      for (OT q = range.first; q < range.second; ++q) {
        const IT row = A.row_id[q];
        IT slot = (row * kHashScale) & mask;
        while (table[slot].key != kEmpty && table[slot].key != row) slot = (slot + 1) & mask;
        table[slot].key = row;
        table[slot].value = SemiRing::Add(SemiRing::Multiply(A.val[q], B.val[p]), table[slot].value);
      }
    }
    IT occupied = 0;
    for (IT slot = 0; slot < capacity; ++slot) {
      if (table[slot].key != kEmpty) table[occupied++] = table[slot];
    }
    std::sort(table.begin(), table.begin() + occupied,
              [](const HashKeyEntry& a, const HashKeyEntry& b) { return a.key < b.key; });
    OT dest = offsets[col];
    for (IT slot = 0; slot < occupied; ++slot) {
      result.entries[dest++] = {table[slot].key, B.col_id[col], table[slot].value};
    }
  }
}

}  // namespace

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

  // @profiler-region hash_lookup start
  const DcscColumnLookup<IT, NT, OT> lookup(A);
  // @profiler-region hash_lookup end
  // @profiler-region hash_estimate_flops start
  const V<OT> flops = EstimateDcscFlops(A, B, lookup);
  // @profiler-region hash_estimate_flops end
  // @profiler-region hash_flops_prefix start
  const V<OT> flop_prefixsum = OmpPrefixSum(flops, threads);
  // @profiler-region hash_flops_prefix end
  // @profiler-region hash_symbolic start
  const V<IT> counts = HashSpGEMMDcscSymbolicPhase(A, B, flop_prefixsum, lookup);
  // @profiler-region hash_symbolic end

  // @profiler-region hash_output_offsets start
  const V<OT> offsets = OmpPrefixSum<IT,OT>(counts, threads);
  // @profiler-region hash_output_offsets end
  // @profiler-region hash_allocate start
  result.Allocate(static_cast<OT>(offsets.back()), A.m, B.n);
  // @profiler-region hash_allocate end
  // @profiler-region hash_numeric start
  HashSpGEMMDcscNumericPhase<SemiRing>(A, B, offsets, lookup, result);
  // @profiler-region hash_numeric end
  return result;
}

#define SPCRAFT_DEFINE_HASH_SPGEMM_DCSC_TYPES(IT, NT, OT)                      \
  template CooMatrix<IT, NT, OT> OmpHashSpGEMM<PlusTimesRing<NT>, IT, NT, OT>( \
      const DcscMatrix<IT, NT, OT>&, const DcscMatrix<IT, NT, OT>&);

#define SPCRAFT_DEFINE_HASH_SPGEMM_DCSC(r, product)                                          \
  SPCRAFT_DEFINE_HASH_SPGEMM_DCSC_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                        BOOST_PP_SEQ_ELEM(1, product),                       \
                                        BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// Pair the storage types so 64-bit indices never select 32-bit offsets.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_HASH_SPGEMM_DCSC,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)))
// clang-format on
#undef SPCRAFT_DEFINE_HASH_SPGEMM_DCSC
#undef SPCRAFT_DEFINE_HASH_SPGEMM_DCSC_TYPES

}  // namespace spcraft
