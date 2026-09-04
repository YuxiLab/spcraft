#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/CsrMatrix.h"

namespace spcraft
{

template <class IT, class NT, class OT>
std::tuple<OT*, IT*, NT*> CsrMatrix<IT, NT, OT>::SafeAllocate(IT m, OT nnz)
{
  if constexpr (std::is_signed_v<IT>) {
    if (m < 0) throw std::invalid_argument("CsrMatrix row count must be non-negative");
  }
  if constexpr (std::is_signed_v<OT>) {
    if (nnz < 0) throw std::invalid_argument("CsrMatrix nnz must be non-negative");
  }

  OT* r = static_cast<OT*>(std::calloc(static_cast<std::size_t>(m) + 1, sizeof(OT)));
  IT* c = nullptr;
  NT* v = nullptr;
  if (nnz != OT{0}) {
    c = static_cast<IT*>(std::calloc(static_cast<std::size_t>(nnz), sizeof(IT)));
    v = static_cast<NT*>(std::calloc(static_cast<std::size_t>(nnz), sizeof(NT)));
  }
  if (r == nullptr || (nnz != OT{0} && (c == nullptr || v == nullptr))) {
    std::free(r);
    std::free(c);
    std::free(v);
    throw std::bad_alloc();
  }
  return std::make_tuple(r, c, v);
}

template <class IT, class NT, class OT>
void CsrMatrix<IT, NT, OT>::SafeDelete(bool memowned, OT* row_ptr, IT* col_id, NT* val)
{
  if (memowned) {
    std::free(row_ptr);
    row_ptr = nullptr;
    std::free(col_id);
    col_id = nullptr;
    std::free(val);
    val = nullptr;
  }
}

template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT>::CsrMatrix(OT* row_ptr_, IT* col_id_, NT* val_, OT nnz_, IT m_, IT n_)
    : row_ptr(row_ptr_), col_id(col_id_), val(val_), nnz(nnz_), m(m_), n(n_), memowned(false)
{
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
  rhs.Reset();
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

template <class IT, class NT, class OT>
void CsrMatrix<IT, NT, OT>::Allocate(OT require_nnz, IT nRows, IT nCols)
{
  if constexpr (std::is_signed_v<IT>) {
    if (nRows < 0 || nCols < 0) {
      throw std::invalid_argument("CsrMatrix dimensions must be non-negative");
    }
  }
  if constexpr (std::is_signed_v<OT>) {
    if (require_nnz < 0) {
      throw std::invalid_argument("CsrMatrix nnz must be non-negative");
    }
  }

  auto replacement = SafeAllocate(nRows, require_nnz);
  SafeDelete(memowned, row_ptr, col_id, val);
  m = nRows;
  n = nCols;
  nnz = require_nnz;
  memowned = true;
  std::tie(row_ptr, col_id, val) = replacement;
}

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
void CsrMatrix<IT, NT, OT>::Reset()
{
  row_ptr = nullptr;
  col_id = nullptr;
  val = nullptr;
  nnz = 0;
  m = 0;
  n = 0;
  memowned = false;
}

}  // namespace spcraft
