#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <fast_matrix_market/fast_matrix_market.hpp>
#include "utils/omp/omp_wrapper.h"
#include "core/AllocationGuard.h"
#include "core/CooMatrix.h"
#include "core/CsrMatrix.h"

namespace spcraft
{
/*************************************
 *             constructor
 *************************************/

template <class IT, class NT, class OT>
CooMatrix<IT, NT, OT>& CooMatrix<IT, NT, OT>::operator=(CooMatrix<IT, NT, OT>&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, entries);
    entries = rhs.entries;
    nnz = rhs.nnz;
    m = rhs.m;
    n = rhs.n;
    memowned = rhs.memowned;
    rhs.Reset();
  }
  return *this;
}

template <class IT, class NT, class OT>
CooMatrix<IT, NT, OT>::CooMatrix(const std::string& filename)
{
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open COO matrix file: " + filename);
  }
  fast_matrix_market::matrix_market_header header;
  fast_matrix_market::read_header(ifs, header);
  // clang-format off
    if (!std::in_range<IT>(header.nrows) || !std::in_range<IT>(header.ncols)) throw std::overflow_error("Matrix Market dimensions exceed COO index range");
  // clang-format on
  std::vector<IT> r_vec;
  std::vector<IT> c_vec;
  std::vector<NT> v_vec;
  // Temporary parser buffers are copied into the single owned Entry array.
  fast_matrix_market::read_matrix_market_body_triplet(
      ifs, header, r_vec, c_vec, v_vec,
      fast_matrix_market::pattern_default_value(static_cast<const NT*>(nullptr)),
      fast_matrix_market::read_options{});
  // clang-format off
    if (!std::in_range<OT>(r_vec.size())) throw std::overflow_error("Matrix Market nonzeros exceed COO offset range");
  // clang-format on
  CooMatrix<IT, NT, OT> coo;
  coo.Allocate(r_vec.size(), header.nrows, header.ncols);
  OMP_PARALLEL_FOR()
  for (auto i = 0; i < r_vec.size(); ++i) coo.entries[i] = {r_vec[i], c_vec[i], v_vec[i]};
  *this = std::move(coo);
}

/*************************************
 *             member functions
 *************************************/

template <class IT, class NT, class OT>
void CooMatrix<IT, NT, OT>::Allocate(OT count, IT rows, IT columns)
{
  Entry* replacement = SafeAllocate(count, rows, columns);
  SafeDelete(memowned, entries);
  entries = replacement;
  nnz = count;
  m = rows;
  n = columns;
  memowned = true;
}

template <class IT, class NT, class OT>
CooMatrix<IT, NT, OT> CooMatrix<IT, NT, OT>::Clone() const
{
  CooMatrix ret;
  ret.Allocate(nnz, m, n);
  if (nnz > 0) {
    OMP_PARALLEL_FOR()
    for (auto i = 0; i < nnz; ++i) {
      ret.entries[i] = entries[i];
    }
  }
  return ret;
}

template <class IT, class NT, class OT>
void CooMatrix<IT, NT, OT>::Reset() noexcept
{
  entries = nullptr;
  nnz = 0;
  m = n = 0;
  memowned = false;
}

template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT> CooMatrix<IT, NT, OT>::ToCsr() const
{
  CsrMatrix<IT, NT, OT> csr(nnz, m, n);
  if (nnz == 0 || m == 0 || n == 0) return csr;
  // loop nnz, check each nnz quality
  int invalid_data = 0;
  OMP_PARALLEL_FOR(schedule(static, 32) reduction(| : invalid_data))
  for (OT i = 0; i < nnz; ++i) {
    const auto& [r, c, value] = entries[i];
    invalid_data |= (r < 0 || c < 0 || r >= m || c >= n);
  }
  if (invalid_data) {
    throw std::invalid_argument("CooMatrix::ToCsr index is out of range");
  }
  // allocate memory
  std::fill(csr.row_ptr, csr.row_ptr + m + 1, 0);
  // create row ptr - step 1: counting nnz per row
  OMP_PARALLEL_FOR(schedule(static, 32))
  for (OT i = 0; i < nnz; ++i) {
    const auto& [r, c, value] = entries[i];
    OMP_ATOMIC
    csr.row_ptr[r + 1]++;
  }
  // create row ptr - step 2: prefix sum
  // TODO: parallize this step.
  for (IT row = 0; row < m; ++row) {
    csr.row_ptr[row + 1] += csr.row_ptr[row];
  }
  // create col_id and val: step 1: put col id and value inside each row (no order)
  std::vector<OT> next(csr.row_ptr, csr.row_ptr + m);
  OMP_PARALLEL_FOR(schedule(static, 32))
  for (OT i = 0; i < nnz; ++i) {
    const auto& [r, c, value] = entries[i];
    OT pos;
    OMP_ATOMIC_CAPTURE
    pos = next[r]++;  // atomic is necessary here.
    csr.col_id[pos] = c;
    csr.val[pos] = value;
  }
  // create col_id and val: step 2: sort col_id and val per row.
  OMP_PARALLEL_FOR(schedule(dynamic))
  for (IT row = 0; row < csr.m; ++row) {
    // get begin pos and size of current row
    const OT begin = csr.row_ptr[row];
    const OT size = csr.row_ptr[row + 1] - begin;
    if (size < 2) continue;  // early exit
    // workspace vars
    std::vector<OT> order(size);
    std::vector<IT> col_id_sorted(size);
    std::vector<NT> val_sorted(size);
    std::iota(order.begin(), order.end(), OT{0});
    // sorting based on col_id
    std::sort(order.begin(), order.end(),
              [&](OT a, OT b) { return csr.col_id[begin + a] < csr.col_id[begin + b]; });
    // store sorted col_id and val.
    for (OT j = 0; j < order.size(); j++) {
      col_id_sorted[j] = csr.col_id[begin + order[j]];
      val_sorted[j] = csr.val[begin + order[j]];
    }
    // copy back
    std::copy(col_id_sorted.begin(), col_id_sorted.end(), csr.col_id + begin);
    std::copy(val_sorted.begin(), val_sorted.end(), csr.val + begin);
  }
  return csr;
}

/*************************************
 *             Static Members
 *************************************/

template <class IT, class NT, class OT>
typename CooMatrix<IT, NT, OT>::Entry* CooMatrix<IT, NT, OT>::SafeAllocate(OT count, IT rows,
                                                                           IT columns)
{
  static_assert(std::is_trivially_copyable_v<Entry>,
                "CooMatrix entries must be trivially copyable");
  static_assert(std::is_trivially_destructible_v<Entry>,
                "CooMatrix entries must be trivially destructible");
  static_assert(alignof(Entry) <= alignof(std::max_align_t),
                "CooMatrix malloc storage does not support over-aligned entries");

  if constexpr (std::is_signed_v<IT>) {
    if (rows < 0 || columns < 0) {
      throw std::invalid_argument("CooMatrix dimensions must be non-negative");
    }
  }
  if constexpr (std::is_signed_v<OT>) {
    if (count < 0) {
      throw std::invalid_argument("CooMatrix nnz must be non-negative");
    }
  }
  if ((rows == 0 || columns == 0) && count != OT{0}) {
    throw std::invalid_argument("nonempty CooMatrix requires nonzero dimensions");
  }

  const std::size_t entry_count = detail::CheckedElementCount<Entry>(count);
  if (entry_count == 0) return nullptr;

  Entry* result = static_cast<Entry*>(std::malloc(entry_count * sizeof(Entry)));
  if (result == nullptr) throw std::bad_alloc();
  return result;
}

template <class IT, class NT, class OT>
void CooMatrix<IT, NT, OT>::SafeDelete(bool owned, Entry* entries) noexcept
{
  if (!owned || entries == nullptr) return;
  std::free(entries);
}
}  // namespace spcraft
