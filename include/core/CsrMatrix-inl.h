#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <iostream>

#include <fmt/format.h>
#include "core/CsrMatrix.h"
#include "core/AllocationGuard.h"

namespace spcraft
{

/*************************************
 *             constructor
 *************************************/
template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT>::CsrMatrix(OT nnz_, IT m_, IT n_) : nnz(nnz_), m(m_), n(n_)
{
  Allocate(nnz, m, n);
}

template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT>::CsrMatrix(CsrMatrix<IT, NT, OT>&& rhs) noexcept
    : row_ptr(rhs.row_ptr),
      col_id(rhs.col_id),
      val(rhs.val),
      nnz(rhs.nnz),
      m(rhs.m),
      n(rhs.n),
      memowned(rhs.memowned)
{
  rhs.memowned = false;
  rhs.row_ptr = nullptr;
  rhs.col_id = nullptr;
  rhs.val = nullptr;
  rhs.nnz = 0;
  rhs.m = 0;
  rhs.n = 0;
}

template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT>& CsrMatrix<IT, NT, OT>::operator=(CsrMatrix<IT, NT, OT>&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, row_ptr, col_id, val);
    row_ptr = rhs.row_ptr;
    col_id = rhs.col_id;
    val = rhs.val;
    nnz = rhs.nnz;
    m = rhs.m;
    n = rhs.n;
    memowned = rhs.memowned;
    rhs.Reset();
  }
  return *this;
}

template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT>::~CsrMatrix()
{
  SafeDelete(memowned, row_ptr, col_id, val);
}

/*************************************
 *             member functions
 *************************************/

template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT> CsrMatrix<IT, NT, OT>::Clone() const
{
  CsrMatrix<IT, NT, OT> res;
  res.Allocate(nnz, m, n);
  if (row_ptr) {
    std::copy(row_ptr, row_ptr + m + 1, res.row_ptr);
  }
  if (col_id) {
    std::copy(col_id, col_id + nnz, res.col_id);
  }
  if (val) {
    std::copy(val, val + nnz, res.val);
  }
  return res;
}

template <class IT, class NT, class OT>
void CsrMatrix<IT, NT, OT>::Allocate(OT require_nnz, IT nRows, IT nCols)
{
  auto replacement = SafeAllocate(require_nnz, nRows, nCols);
  SafeDelete(memowned, row_ptr, col_id, val);
  std::tie(row_ptr, col_id, val) = replacement;
  nnz = require_nnz;
  m = nRows;
  n = nCols;
  memowned = true;
}

template <class IT, class NT, class OT>
void CsrMatrix<IT, NT, OT>::Reset() noexcept
{
  row_ptr = nullptr;
  col_id = nullptr;
  val = nullptr;
  nnz = 0;
  m = 0;
  n = 0;
  memowned = false;
}

/*************************************
 *             Static Members
 *************************************/
template <class IT, class NT, class OT>
std::tuple<OT*, IT*, NT*> CsrMatrix<IT, NT, OT>::SafeAllocate(OT required_nnz, IT rows, IT columns)
{
  // if IT is signed type, chk row,col >= 0
  if constexpr (std::is_signed_v<IT>) {
    if (rows < 0 || columns < 0) {
      throw std::invalid_argument("CsrMatrix dimensions must be non-negative");
    }
  }
  // if OT is signed type, chk nnz is >= 0
  if constexpr (std::is_signed_v<OT>) {
    if (required_nnz < 0) {
      throw std::invalid_argument("CsrMatrix nnz must be non-negative");
    }
  }
  // row and col should be >=0 if nnz is >= 0
  if ((rows == 0 || columns == 0) && required_nnz != OT{0}) {
    throw std::invalid_argument("nonempty CsrMatrix requires nonzero dimensions");
  }

  // Calculate every size before allocating anything.
  const std::size_t rows_size = detail::CheckedElementCount<OT>(rows);
  // if row_size IS max_count, we need row_size + 1 to not overflow.
  const std::size_t max_count = std::numeric_limits<std::size_t>::max() / sizeof(OT);
  if (rows_size >= max_count) {
    throw std::bad_array_new_length();
  }
  const std::size_t row_ptr_count = rows_size + 1;
  const std::size_t index_count = detail::CheckedElementCount<IT>(required_nnz);
  const std::size_t value_count = detail::CheckedElementCount<NT>(required_nnz);

  OT* new_row_ptr = static_cast<OT*>(std::malloc(row_ptr_count * sizeof(OT)));
  IT* new_col_id = nullptr;
  NT* new_val = nullptr;
  if (required_nnz != OT{0}) {
    new_col_id = static_cast<IT*>(std::malloc(index_count * sizeof(IT)));
    new_val = static_cast<NT*>(std::malloc(value_count * sizeof(NT)));
  }
  // make sure 3 arraies are all allocated successfully.
  if (new_row_ptr == nullptr ||
      (required_nnz != OT{0} && (new_col_id == nullptr || new_val == nullptr))) {
    std::free(new_row_ptr);
    std::free(new_col_id);
    std::free(new_val);
    throw std::bad_alloc();
  }

  return {new_row_ptr, new_col_id, new_val};
}

template <class IT, class NT, class OT>
void CsrMatrix<IT, NT, OT>::SafeDelete(bool owned, OT* row_ptr, IT* col_id, NT* val) noexcept
{
  if (!owned) return;
  std::free(row_ptr);
  std::free(col_id);
  std::free(val);
}

}  // namespace spcraft
