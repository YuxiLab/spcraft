#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/CooMatrix.h"
#include "core/DcscMatrix.h"
#include "kernel/SpGEMMHash.h"
#include "semiring/SemiRing.h"

namespace spcraft
{

/**
 * @brief Source-mapped from CombBLAS LocalSpGEMMHash.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] CooMatrix<IT, NT, OT> OmpHashSpGEMM(const DcscMatrix<IT, NT, OT>& A,
                                                  const DcscMatrix<IT, NT, OT>& B);

/**
 * @brief Reference 1 level Omp SpGEMM.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] CooMatrix<IT, NT, OT> OmpSpGEMM(const CscMatrix<IT, NT, OT>& A,
                                              const CscMatrix<IT, NT, OT>& B);

// Checked arithmetic is the only change to the two-static-loop prefix algorithm.
template <class OT>
OT CombBLASCheckedSum(OT a, OT b, int& overflow)
{
  if (b > std::numeric_limits<OT>::max() - a) {
    overflow = 1;
    return std::numeric_limits<OT>::max();
  }
  return a + b;
}

template <class OT>
std::vector<OT> OmpPrefixSum(const std::vector<OT>& in, int threads)
{
  const auto size = in.size();
  std::vector<OT> out(size + 1);
  std::vector<OT> tsum(threads + 1);
  int overflow = 0;
  OMP_PARALLEL(num_threads(threads) default(none) shared(in, out, tsum, size, overflow))
  {
    int thread = 0;
    thread = OMP_GET_THREAD_NUM();
    OT sum = 0;
    int local_overflow = 0;
    // First, sum each thread's own block.
    OMP_FOR(schedule(static))
    for (std::size_t i = 0; i < size; ++i) {
      sum = CombBLASCheckedSum(sum, in[i], local_overflow);
      out[i + 1] = sum;
    }
    tsum[thread + 1] = sum;
    OMP_BARRIER
    OT offset = 0;
    // Then add the totals from earlier blocks to get global offsets.
    for (int i = 0; i < thread + 1; ++i) {
      offset = CombBLASCheckedSum(offset, tsum[i], local_overflow);
    }
    OMP_FOR(schedule(static))
    for (std::size_t i = 0; i < size; ++i) {
      out[i + 1] = CombBLASCheckedSum(out[i + 1], offset, local_overflow);
    }
    if (local_overflow) {
      OMP_CRITICAL
      overflow = 1;
    }
  }
  if (overflow) {
    throw std::overflow_error("CombBLAS-mapped prefix sum exceeds offset type limit");
  }
  return out;
}

template <class IT, class NT, class OT>
class CombBLASColumnLookup
{
  const DcscMatrix<IT, NT, OT>& matrix;

 public:
  IT csize = 0;
  std::vector<IT> aux;
  explicit CombBLASColumnLookup(const DcscMatrix<IT, NT, OT>& a) : matrix(a)
  {
    // Preserve upstream's float arithmetic on a range where its integer round
    // trips are exact. DCSC storage itself supports larger logical shapes.
    constexpr std::uintmax_t kFloatExactIntegerLimit = std::uintmax_t{1} << 24;
    const auto width = static_cast<std::uintmax_t>(a.n);
    if (width >= kFloatExactIntegerLimit ||
        width >= static_cast<std::uintmax_t>(std::numeric_limits<IT>::max()))
      throw std::overflow_error("CombBLAS-mapped lookup extent exceeds supported reference range");
    const auto extent = width + 1;
    // Split logical columns into chunks to narrow later column searches.
    const float cf = static_cast<float>(extent) / static_cast<float>(a.nzc);
    csize = static_cast<IT>(std::ceil(cf));
    const IT chunks = static_cast<IT>(std::ceil(static_cast<float>(extent) / std::ceil(cf)));
    aux.resize(CheckedElementCount<IT>(static_cast<std::uintmax_t>(chunks) + 1));
    IT reg = 0, current = 0;
    aux[current++] = 0;
    for (IT i = 0; i < a.nzc; ++i) {
      // Widen the boundary product to avoid the upstream signed-overflow edge.
      while (static_cast<std::uintmax_t>(a.col_id[i]) >=
             static_cast<std::uintmax_t>(current) * csize)
        aux[current++] = reg;
      reg = i + 1;
    }
    while (current <= chunks) aux[current++] = reg;
  }
  void FillColInds(const IT* columns, std::size_t count,
                   std::vector<std::pair<OT, OT>>& ranges) const
  {
    constexpr std::size_t kScanningThreshold = 4;  // CombBLAS SpDefs.h THRESHOLD.
    if (!count) return;
    if (static_cast<std::size_t>(matrix.nzc) / count < kScanningThreshold) {
      // For many requested columns, intersect the two sorted column lists.
      using IndexPair = std::pair<IT, IT>;
      std::vector<IndexPair> intersection(std::min(static_cast<std::size_t>(matrix.nzc), count));
      std::vector<IndexPair> range1(matrix.nzc), range2(count);
      for (IT i = 0; i < matrix.nzc; ++i) range1[i] = {matrix.col_id[i], i};
      for (std::size_t i = 0; i < count; ++i) range2[i] = {columns[i], 0};
      const auto end = std::set_intersection(
          range1.begin(), range1.end(), range2.begin(), range2.end(), intersection.begin(),
          [](const IndexPair& a, const IndexPair& b) { return a.first < b.first; });
      std::size_t i = 0;
      const auto matches = static_cast<std::size_t>(end - intersection.begin());
      for (std::size_t j = 0; j < count; ++j) {
        if (i == matches || intersection[i].first != columns[j])
          ranges[j] = {0, 0};
        else {
          const IT slot = intersection[i++].second;
          ranges[j] = {matrix.col_ptr[slot], matrix.col_ptr[slot + 1]};
        }
      }
    } else {
      // For a few requested columns, search only their lookup chunks.
      for (std::size_t j = 0; j < count; ++j) {
        const IT base = static_cast<IT>(std::floor(static_cast<float>(columns[j] / csize)));
        const auto* begin = matrix.col_id + aux[base];
        const auto* end = matrix.col_id + aux[base + 1];
        const auto* found = std::find(begin, end, columns[j]);
        if (found == end)
          ranges[j] = {0, 0};
        else {
          const auto slot = found - matrix.col_id;
          ranges[j] = {matrix.col_ptr[slot], matrix.col_ptr[slot + 1]};
        }
      }
    }
  }
};

// Upstream allocates at least 16 slots even for a zero-work/output column.
template <class OT>
std::size_t CombBLASHashCapacity(OT count)
{
  constexpr std::size_t kMinHashTableSize = 16;
  std::size_t capacity = kMinHashTableSize;
  while (capacity < static_cast<std::uintmax_t>(count)) capacity <<= 1;
  return capacity;
}

// A serial maximum-size preflight keeps sizing exceptions out of worker loops;
// each worker still calculates its capacity from flop/colptr exactly as upstream.
template <class Entry, class OT>
void CombBLASValidateCapacities(const std::vector<OT>& counts)
{
  const auto maximum = counts.empty() ? OT{0} : *std::max_element(counts.begin(), counts.end());
  (void)SpGEMMHashCapacity<Entry>(std::max<std::uintmax_t>(1, maximum));
}

template <class IT, class NT, class OT>
std::vector<OT> CombBLASEstimateFLOP(const DcscMatrix<IT, NT, OT>& A,
                                     const DcscMatrix<IT, NT, OT>& B)
{
  const CombBLASColumnLookup<IT, NT, OT> lookup(A);
  const int threads = OMP_GET_NUM_THREADS();
  std::vector<OT> flop(B.nzc);
  OMP_PARALLEL_FOR()
  for (IT i = 0; i < B.nzc; ++i) flop[i] = 0;
  std::vector<std::vector<std::pair<OT, OT>>> colinds(threads);
  OMP_PARALLEL_FOR()
  for (IT i = 0; i < B.nzc; ++i) {
    int thread = 0;
    thread = OMP_GET_THREAD_NUM();
    const auto count = static_cast<std::size_t>(B.col_ptr[i + 1] - B.col_ptr[i]);
    auto& ranges = colinds[thread];
    if (ranges.size() < count) ranges.resize(count);
    lookup.FillColInds(B.row_id + B.col_ptr[i], count, ranges);
    for (std::size_t j = 0; j < count; ++j) {
      flop[i] += ranges[j].second - ranges[j].first;
    }
  }
  return flop;
}

template <class IT, class NT, class OT>
std::vector<OT> CombBLASEstimateNNZHash(const DcscMatrix<IT, NT, OT>& A,
                                        const DcscMatrix<IT, NT, OT>& B,
                                        const std::vector<OT>& flop)
{
  const CombBLASColumnLookup<IT, NT, OT> lookup(A);
  const int threads = OMP_GET_NUM_THREADS();
  // Symbolic phase: count distinct output rows without computing values.
  CombBLASValidateCapacities<IT>(flop);
  std::vector<OT> counts(CheckedElementCount<OT>(B.nzc));
  std::vector<std::vector<std::pair<OT, OT>>> colinds(threads);
  std::vector<std::vector<IT>> hash(threads);
  OMP_PARALLEL_FOR(num_threads(threads) default(none)
                       shared(A, B, lookup, flop, counts, colinds, hash))
  for (IT i = 0; i < B.nzc; ++i) {
    counts[i] = 0;
    int thread = 0;
    thread = OMP_GET_THREAD_NUM();
    const auto count = static_cast<std::size_t>(B.col_ptr[i + 1] - B.col_ptr[i]);
    auto& ranges = colinds[thread];
    if (ranges.size() < count) ranges.resize(count);
    lookup.FillColInds(B.row_id + B.col_ptr[i], count, ranges);
    // The product count bounds the number of distinct rows in this column.
    const auto capacity = CombBLASHashCapacity(flop[i]);
    auto& keys = hash[thread];
    if (keys.size() < capacity) keys.resize(capacity);
    std::fill_n(keys.begin(), capacity, IT(-1));
    // Insert each row once; repeated rows share the same output entry.
    for (std::size_t j = 0; j < count; ++j) {
      for (OT k = ranges[j].first; k < ranges[j].second; ++k) {
        const IT key = A.row_id[k];
        auto slot = SpGEMMHashSlot(key, capacity - 1);
        while (true) {
          if (keys[slot] == key) break;
          if (keys[slot] == IT(-1)) {
            keys[slot] = key;
            ++counts[i];
            break;
          }
          // Resolve a collision by probing the next slot, wrapping at the end.
          slot = (slot + 1) & (capacity - 1);
        }
      }
    }
  }
  return counts;
}

/**
 * @brief Source-mapped CombBLAS LocalSpGEMMHash(A,B,false,false,true) baseline.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] CooMatrix<IT, NT, OT> OmpHashSpGEMM(const DcscMatrix<IT, NT, OT>& A,
                                                  const DcscMatrix<IT, NT, OT>& B)
{
  // clang-format off
  if (A.n != B.m) throw std::invalid_argument("SpGEMM matrix dimensions do not match");
  if (A.m < 0 || A.n < 0 || B.m < 0 || B.n < 0) throw std::invalid_argument("SpGEMM dimensions must be nonnegative");
  // clang-format on
  CooMatrix<IT, NT, OT> result;
  if (!A.nnz || !B.nnz) {
    result.Allocate(0, A.m, B.n);
    return result;
  }
  // 2. Build A's column lookup and count scalar products per output column.
  const CombBLASColumnLookup<IT, NT, OT> lookup(A);
  const int threads = OMP_GET_NUM_THREADS();
  auto flop = CombBLASEstimateFLOP(A, B);
  // Keep this work prefix sum to match the CombBLAS baseline.
  const auto flopptr = OmpPrefixSum(flop, threads);
  // 3. Symbolic phase: count unique rows and assign each column a COO slice.
  auto counts = CombBLASEstimateNNZHash(A, B, flop);
  const auto offsets = OmpPrefixSum(counts, threads);
  using Entry = std::pair<IT, NT>;
  CombBLASValidateCapacities<Entry>(counts);
  // Release the same phase-local arrays that upstream frees before numeric work.
  std::vector<OT>().swap(counts);
  std::vector<OT>().swap(flop);
  // 4. Allocate the exact output size and prepare column-range scratch.
  result.Allocate(offsets.back(), A.m, B.n);
  std::vector<std::vector<std::pair<OT, OT>>> colinds(threads);
  const auto initial = CheckedElementCount<std::pair<OT, OT>>(A.nnz / threads);
  for (int i = 0; i < threads; ++i) colinds[i].resize(initial);
  // 5. Numeric phase: compute each stored column of B independently.
  OMP_PARALLEL_FOR()
  for (IT i = 0; i < B.nzc; ++i) {
    int thread = 0;
    thread = OMP_GET_THREAD_NUM();
    const auto count = static_cast<std::size_t>(B.col_ptr[i + 1] - B.col_ptr[i]);
    auto& ranges = colinds[thread];
    if (ranges.size() < count) ranges.resize(count);
    // Find the A columns selected by this B column's row indices.
    lookup.FillColInds(B.row_id + B.col_ptr[i], count, ranges);
    const auto capacity = CombBLASHashCapacity(static_cast<OT>(offsets[i + 1] - offsets[i]));
    std::vector<Entry> table(capacity);
    // Each hash slot holds an output row and its accumulated value.
    for (std::size_t j = 0; j < capacity; ++j) table[j].first = IT(-1);
    // Expand A(:, k) * B(k, j) and merge products with the same output row.
    for (std::size_t j = 0; j < count; ++j) {
      const NT bval = B.val[B.col_ptr[i] + j];
      for (OT k = ranges[j].first; k < ranges[j].second; ++k) {
        const NT product = SemiRing::Multiply(A.val[k], bval);
        const IT key = A.row_id[k];
        auto slot = SpGEMMHashSlot(key, capacity - 1);
        while (true) {
          if (table[slot].first == key) {
            table[slot].second = SemiRing::Add(product, table[slot].second);
            break;
          }
          if (table[slot].first == IT(-1)) {
            // The first product initializes the value, as in CombBLAS.
            table[slot].first = key;
            table[slot].second = product;
            break;
          }
          // Linear probing finds either this row or the next empty slot.
          slot = (slot + 1) & (capacity - 1);
        }
      }
    }
    // 6. Compact occupied slots and sort this column by row index.
    std::size_t occupied = 0;
    for (std::size_t j = 0; j < capacity; ++j)
      if (table[j].first != IT(-1)) table[occupied++] = table[j];
    std::sort(table.begin(), table.begin() + occupied,
              [](const Entry& a, const Entry& b) { return a.first < b.first; });
    // 7. Write (row, column, value) tuples into this column's reserved slice.
    OT dest = offsets[i];
    for (std::size_t j = 0; j < occupied; ++j)
      result.entries[dest++] = {table[j].first, B.col_id[i], table[j].second};
  }
  return result;
}

}  // namespace spcraft
