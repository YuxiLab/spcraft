#include "numeric/mtMultiwayMerge.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "semiring/SemiRing.h"
#include "utils/omp/omp_wrapper.h"

namespace spcraft
{
namespace
{

template <class OT>
using SplitOffsets = std::vector<std::vector<OT>>;

// Both passes traverse the same heap order without copying the input tuples.
template <class IT, class NT, class OT, class Function>
void VisitSplit(const std::vector<SPCOO>& inputs, const SplitOffsets<OT>& bounds, int64_t split,
                Function visit)
{
  std::vector<OT> positions(inputs.size());
  std::vector<std::size_t> heap;
  heap.reserve(inputs.size());
  for (std::size_t source = 0; source < inputs.size(); ++source) {
    positions[source] = bounds[source][split];
    if (positions[source] < bounds[source][split + 1]) heap.push_back(source);
  }

  const auto later = [&](std::size_t left, std::size_t right) {
    const TupleEntry<IT, NT>& a = inputs[left].entries[positions[left]];
    const TupleEntry<IT, NT>& b = inputs[right].entries[positions[right]];
    if (a.col != b.col) return a.col > b.col;
    if (a.row != b.row) return a.row > b.row;
    // Equal coordinates follow input order, independent of the thread schedule.
    return left > right;
  };
  std::make_heap(heap.begin(), heap.end(), later);
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), later);
    const std::size_t source = heap.back();
    visit(inputs[source].entries[positions[source]]);
    ++positions[source];
    if (positions[source] < bounds[source][split + 1]) {
      std::push_heap(heap.begin(), heap.end(), later);
    } else {
      heap.pop_back();
    }
  }
}

}  // namespace

SP_SR_MAT_TEMP
SPND SPCOO OmpMultiwayMerge(const std::vector<SPCOO>& inputs, IT m, IT n)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");

  // Validate the common shape before allocating output or entering parallel work.
  if constexpr (std::is_signed_v<IT>) {
    if (m < 0 || n < 0) {
      throw std::invalid_argument("Multiway merge dimensions must be nonnegative");
    }
  }
  bool has_entries = false;
  for (const SPCOO& input : inputs) {
    if (input.m != m || input.n != n) {
      throw std::invalid_argument("Multiway merge inputs must have matching dimensions");
    }
    has_entries = has_entries || input.nnz != 0;
  }
  SPCOO result;
  if (!has_entries) {
    result.Allocate(OT{0}, m, n);
    return result;
  }

  // Oversplit columns so dynamic scheduling can balance irregular merge work.
  constexpr int64_t kSplitsPerThread = 4;
  const int64_t desired_splits = kSplitsPerThread * OMP_GET_MAX_THREADS();
  const int64_t splits = std::min(int64_t{n}, desired_splits);
  const int64_t columns_per_split = n / splits;
  const int64_t extra_columns = n % splits;
  SplitOffsets<OT> boundaries(inputs.size(), std::vector<OT>(splits + 1));
  OMP_PARALLEL_FOR()
  for (std::size_t source = 0; source < inputs.size(); ++source) {
    const SPCOO& input = inputs[source];
    boundaries[source][splits] = input.nnz;
    if (input.nnz == 0) continue;
    const TupleEntry<IT, NT>* first = input.entries;
    const TupleEntry<IT, NT>* end = input.entries + input.nnz;
    for (int64_t split = 1; split < splits; ++split) {
      const IT column = split * columns_per_split + std::min(split, extra_columns);
      const auto before_column = [](const TupleEntry<IT, NT>& entry, IT col) { return entry.col < col; };
      first = std::lower_bound(first, end, column, before_column);
      boundaries[source][split] = (OT)(first - input.entries);
    }
  }

  // Count distinct coordinates so allocation depends on output nnz, not input nnz.
  std::vector<int64_t> counts(splits);
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (int64_t split = 0; split < splits; ++split) {
    int64_t count = 0;
    IT previous_row = 0;
    IT previous_col = 0;
    const auto count_entry = [&](const TupleEntry<IT, NT>& entry) {
      if (count == 0 || entry.row != previous_row || entry.col != previous_col) {
        ++count;
        previous_row = entry.row;
        previous_col = entry.col;
      }
    };
    VisitSplit(inputs, boundaries, split, count_entry);
    counts[split] = count;
  }

  // Prefixes stay wide until the final merged size has passed the storage check.
  std::vector<int64_t> offsets(splits + 1);
  for (int64_t split = 0; split < splits; ++split) {
    offsets[split + 1] = offsets[split] + counts[split];
  }
  const int64_t total = offsets[splits];
  if (total > std::numeric_limits<OT>::max()) {
    throw std::overflow_error("Multiway merge output nnz exceeds offset type limit");
  }
  result.Allocate((OT)total, m, n);

  // Whole columns belong to one split, so threads write disjoint output ranges.
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (int64_t split = 0; split < splits; ++split) {
    const OT begin = offsets[split];
    OT position = begin;
    const auto merge_entry = [&](const TupleEntry<IT, NT>& entry) {
      const bool new_coordinate = position == begin || result.entries[position - 1].row != entry.row ||
                                  result.entries[position - 1].col != entry.col;
      if (new_coordinate) {
        result.entries[position++] = {entry.row, entry.col, SemiRing::kAdditiveIdentity};
      }
      NT& value = result.entries[position - 1].val;
      value = SemiRing::Add(value, entry.val);
    };
    VisitSplit(inputs, boundaries, split, merge_entry);
  }
  return result;
}

#define SPCRAFT_DEFINE_MULTIWAY_MERGE_TYPES(IT, NT, OT)                           \
  template CooMatrix<IT, NT, OT> OmpMultiwayMerge<PlusTimesRing<NT>, IT, NT, OT>( \
      const std::vector<CooMatrix<IT, NT, OT>>&, IT, IT);

#define SPCRAFT_DEFINE_MULTIWAY_MERGE(r, product)                                          \
  SPCRAFT_DEFINE_MULTIWAY_MERGE_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                      BOOST_PP_SEQ_ELEM(1, product),                       \
                                      BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_MULTIWAY_MERGE,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)(bool)))
// clang-format on
#undef SPCRAFT_DEFINE_MULTIWAY_MERGE
#undef SPCRAFT_DEFINE_MULTIWAY_MERGE_TYPES

}  // namespace spcraft
