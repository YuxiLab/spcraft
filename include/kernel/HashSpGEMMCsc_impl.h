#pragma once

#include "mtSpGEMM.h"

namespace spcraft
{

template <class IT, class NT, class OT>
std::vector<OT> EstimateFLOP(const CscMatrix<IT, NT, OT>& A, const CscMatrix<IT, NT, OT>& B)
{
  std::vector<OT> flops(static_cast<std::size_t>(B.n), OT{0});
  // const CombBLASColumnLookup<IT, NT, OT> lookup(A);
  // const int threads = OMP_GET_MAX_THREADS();
  // std::vector<OT> flop(B.nzc);
  // OMP_PARALLEL_FOR()
  // for (IT i = 0; i < B.nzc; ++i) flop[i] = 0;
  // std::vector<std::vector<std::pair<OT, OT>>> colinds(threads);
  // OMP_PARALLEL_FOR()
  // for (IT i = 0; i < B.nzc; ++i) {
  //   int thread = 0;
  //   thread = OMP_GET_THREAD_NUM();
  //   const auto count = static_cast<std::size_t>(B.col_ptr[i + 1] - B.col_ptr[i]);
  //   auto& ranges = colinds[thread];
  //   if (ranges.size() < count) ranges.resize(count);
  //   lookup.FillColInds(B.row_id + B.col_ptr[i], count, ranges);
  //   for (std::size_t j = 0; j < count; ++j) {
  //     flop[i] += ranges[j].second - ranges[j].first;
  //   }
  // }
  return flops;
}

/**
 * @brief Source-mapped CombBLAS LocalSpGEMMHash(A,B,false,false,true) baseline.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] CooMatrix<IT, NT, OT> OmpHashSpGEMM(const CscMatrix<IT, NT, OT>& A,
                                                  const CscMatrix<IT, NT, OT>& B)
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
  std::cerr << "hello" << std::endl;
  const int threads = OMP_GET_MAX_THREADS();
  auto flop = EstimateFLOP(A, B);
  // // Keep this work prefix sum to match the CombBLAS baseline.
  // const auto flopptr = OmpPrefixSum(flop, threads);
  // // 3. Symbolic phase: count unique rows and assign each column a COO slice.
  // auto counts = CombBLASEstimateNNZHash(A, B, flop);
  // const auto offsets = OmpPrefixSum(counts, threads);
  // using Entry = std::pair<IT, NT>;
  // CombBLASValidateCapacities<Entry>(counts);
  // // Release the same phase-local arrays that upstream frees before numeric work.
  // std::vector<OT>().swap(counts);
  // std::vector<OT>().swap(flop);
  // // 4. Allocate the exact output size and prepare column-range scratch.
  // result.Allocate(offsets.back(), A.m, B.n);
  // std::vector<std::vector<std::pair<OT, OT>>> colinds(threads);
  // const auto initial = detail::CheckedElementCount<std::pair<OT, OT>>(A.nnz / threads);
  // for (int i = 0; i < threads; ++i) colinds[i].resize(initial);
  // // 5. Numeric phase: compute each stored column of B independently.
  // OMP_PARALLEL_FOR()
  // for (IT i = 0; i < B.nzc; ++i) {
  //   int thread = 0;
  //   thread = OMP_GET_THREAD_NUM();
  //   const auto count = static_cast<std::size_t>(B.col_ptr[i + 1] - B.col_ptr[i]);
  //   auto& ranges = colinds[thread];
  //   if (ranges.size() < count) ranges.resize(count);
  //   // Find the A columns selected by this B column's row indices.
  //   lookup.FillColInds(B.row_id + B.col_ptr[i], count, ranges);
  //   const auto capacity = CombBLASHashCapacity(static_cast<OT>(offsets[i + 1] - offsets[i]));
  //   std::vector<Entry> table(capacity);
  //   // Each hash slot holds an output row and its accumulated value.
  //   for (std::size_t j = 0; j < capacity; ++j) table[j].first = IT(-1);
  //   // Expand A(:, k) * B(k, j) and merge products with the same output row.
  //   for (std::size_t j = 0; j < count; ++j) {
  //     const NT bval = B.val[B.col_ptr[i] + j];
  //     for (OT k = ranges[j].first; k < ranges[j].second; ++k) {
  //       const NT product = SemiRing::Multiply(A.val[k], bval);
  //       const IT key = A.row_id[k];
  //       auto slot = detail::SpGEMMHashSlot(key, capacity - 1);
  //       while (true) {
  //         if (table[slot].first == key) {
  //           table[slot].second = SemiRing::Add(product, table[slot].second);
  //           break;
  //         }
  //         if (table[slot].first == IT(-1)) {
  //           // The first product initializes the value, as in CombBLAS.
  //           table[slot].first = key;
  //           table[slot].second = product;
  //           break;
  //         }
  //         // Linear probing finds either this row or the next empty slot.
  //         slot = (slot + 1) & (capacity - 1);
  //       }
  //     }
  //   }
  //   // 6. Compact occupied slots and sort this column by row index.
  //   std::size_t occupied = 0;
  //   for (std::size_t j = 0; j < capacity; ++j)
  //     if (table[j].first != IT(-1)) table[occupied++] = table[j];
  //   std::sort(table.begin(), table.begin() + occupied,
  //             [](const Entry& a, const Entry& b) { return a.first < b.first; });
  //   // 7. Write (row, column, value) tuples into this column's reserved slice.
  //   OT dest = offsets[i];
  //   for (std::size_t j = 0; j < occupied; ++j)
  //     result.entries[dest++] = {table[j].first, B.col_id[i], table[j].second};
  // }
  return result;
}

}  // namespace spcraft
