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
SP_MAT_TEMP
class DcscColumnLookup
{
 public:
  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  explicit DcscColumnLookup(const SPDCSC& A) : matrix(A)
  {
    // Integer ceiling division avoids losing column IDs through float rounding.
    const int64_t extent = static_cast<int64_t>(A.n) + 1;
    chunk_size = (extent - 1) / A.nzc + 1;
    const int64_t chunks = (extent - 1) / chunk_size + 1;
    aux.resize(detail::CheckedElementCount<IT>(chunks + 1));
    int64_t next_chunk = 1;
    for (IT col = 0; col < A.nzc; ++col) {
      const int64_t chunk = A.col_id[col] / chunk_size;
      while (next_chunk <= chunk) aux[next_chunk++] = col;
    }
    while (next_chunk <= chunks) aux[next_chunk++] = A.nzc;
  }

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  std::vector<std::pair<OT, OT>> FillColInds(const SPDCSC& B, IT col) const
  {
    const OT begin = B.col_ptr[col];
    const OT end = B.col_ptr[col + 1];
    const std::size_t count = static_cast<std::size_t>(end - begin);
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
        const int64_t chunk = column / chunk_size;
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
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  const SPDCSC& matrix;
  VIT aux{};
  int64_t chunk_size = 0;
};

SP_MAT_TEMP
VL EstimateDcscFlops(const SPDCSC& A, const SPDCSC& B, const DcscColumnLookup<IT, NT, OT>& lookup)
{
  VL flops(B.nzc, 0);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT col = 0; col < B.nzc; ++col) {
    const std::vector<std::pair<OT, OT>> ranges = lookup.FillColInds(B, col);
    int64_t count = 0;
    for (const std::pair<OT, OT>& range : ranges) count += range.second - range.first;
    flops[col] = count;
  }
  return flops;
}

SP_MAT_TEMP
VL CountDcscEntries(const SPDCSC& A, const SPDCSC& B, const VL& flop_prefixsum,
                    const DcscColumnLookup<IT, NT, OT>& lookup)
{
  VL counts(B.nzc, 0);
  constexpr std::size_t kMinCapacity = 16;
  constexpr std::size_t kHashScale = 107;
  constexpr IT kEmpty = static_cast<IT>(-1);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT col = 0; col < B.nzc; ++col) {
    const int64_t flops = flop_prefixsum[col + 1] - flop_prefixsum[col];
    if (flops == 0) continue;
    std::size_t capacity = kMinCapacity;
    while (capacity < static_cast<std::size_t>(flops)) capacity <<= 1;
    const std::size_t mask = capacity - 1;
    std::vector<IT> keys(capacity, kEmpty);
    const std::vector<std::pair<OT, OT>> ranges = lookup.FillColInds(B, col);
    int64_t count = 0;
    for (const std::pair<OT, OT>& range : ranges) {
      for (OT p = range.first; p < range.second; ++p) {
        const IT row = A.row_id[p];
        std::size_t slot = (static_cast<std::size_t>(row) * kHashScale) & mask;
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
void AccumulateDcscEntries(const SPDCSC& A, const SPDCSC& B, const VL& offsets,
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
  constexpr std::size_t kMinCapacity = 16;
  constexpr std::size_t kHashScale = 107;
  constexpr IT kEmpty = static_cast<IT>(-1);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT col = 0; col < B.nzc; ++col) {
    const int64_t count = offsets[col + 1] - offsets[col];
    if (count == 0) continue;
    std::size_t capacity = kMinCapacity;
    while (capacity < static_cast<std::size_t>(count)) capacity <<= 1;
    const std::size_t mask = capacity - 1;
    std::vector<HashKeyEntry> table(capacity, {kEmpty, SemiRing::kAdditiveIdentity});
    const std::vector<std::pair<OT, OT>> ranges = lookup.FillColInds(B, col);
    const OT begin = B.col_ptr[col];
    const OT end = B.col_ptr[col + 1];
    for (OT p = begin; p < end; ++p) {
      const std::pair<OT, OT>& range = ranges[p - begin];
      for (OT q = range.first; q < range.second; ++q) {
        const IT row = A.row_id[q];
        std::size_t slot = (static_cast<std::size_t>(row) * kHashScale) & mask;
        while (table[slot].key != kEmpty && table[slot].key != row) slot = (slot + 1) & mask;
        table[slot].key = row;
        table[slot].value = SemiRing::Add(SemiRing::Multiply(A.val[q], B.val[p]), table[slot].value);
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

  // TODO: fix SpDCSC HashSpGEMM kernel and do the profiling.
  
  // @profiler-region hash_lookup start
  const DcscColumnLookup<IT, NT, OT> lookup(A);
  // @profiler-region hash_lookup end
  // @profiler-region hash_estimate_flops start
  const VL flops = EstimateDcscFlops(A, B, lookup);
  // @profiler-region hash_estimate_flops end
  // @profiler-region hash_flops_prefix start
  const VL flop_prefixsum = OmpPrefixSum(flops, threads);
  // @profiler-region hash_flops_prefix end
  // @profiler-region hash_symbolic start
  const VL counts = CountDcscEntries(A, B, flop_prefixsum, lookup);
  // @profiler-region hash_symbolic end

  // @profiler-region hash_output_offsets start
  const VL offsets = OmpPrefixSum(counts, threads);
  // @profiler-region hash_output_offsets end
  if (static_cast<std::uintmax_t>(offsets.back()) >
      static_cast<std::uintmax_t>(std::numeric_limits<OT>::max())) {
    throw std::overflow_error("SpGEMM output nnz exceeds offset type limit");
  }
  // @profiler-region hash_allocate start
  result.Allocate(static_cast<OT>(offsets.back()), A.m, B.n);
  // @profiler-region hash_allocate end
  // @profiler-region hash_numeric start
  AccumulateDcscEntries<SemiRing>(A, B, offsets, lookup, result);
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
