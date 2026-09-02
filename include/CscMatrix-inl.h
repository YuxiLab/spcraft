#pragma once

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "CscMatrix.h"

namespace spcraft
{

template <class IT, class NT, class OT>
std::tuple<OT*, IT*, NT*> CscMatrix<IT, NT, OT>::SafeAllocate(IT n, OT nnz)
{
  if (nnz <= 0) throw std::invalid_argument("CscMatrix::SafeAllocate requires nnz > 0");
  OT* c = static_cast<OT*>(std::calloc(n + 1, sizeof(OT)));
  IT* r = static_cast<IT*>(std::calloc(nnz, sizeof(IT)));
  NT* v = static_cast<NT*>(std::calloc(nnz, sizeof(NT)));
  return std::make_tuple(c, r, v);
}

template <class IT, class NT, class OT>
void CscMatrix<IT, NT, OT>::SafeDelete(bool memowned, OT* col_ptr, IT* row_id, NT* val)
{
  if (memowned) {
    std::free(col_ptr); col_ptr = nullptr;
    std::free(row_id); row_id = nullptr;
    std::free(val); val = nullptr;
  }
}

template <class IT, class NT, class OT>
CscMatrix<IT, NT, OT>::CscMatrix(OT* col_ptr_, IT* row_id_, NT* val_, OT nnz_, IT m_, IT n_)
    : col_ptr(col_ptr_), row_id(row_id_), val(val_), nnz(nnz_), m(m_), n(n_), memowned(false)
{
}

template <class IT, class NT, class OT>
CscMatrix<IT, NT, OT>::CscMatrix(CscMatrix<IT, NT, OT>&& rhs) noexcept
    : col_ptr(rhs.col_ptr),
      row_id(rhs.row_id),
      val(rhs.val),
      nnz(rhs.nnz),
      m(rhs.m),
      n(rhs.n),
      memowned(rhs.memowned)
{
  rhs.Reset();
}

template <class IT, class NT, class OT>
CscMatrix<IT, NT, OT>& CscMatrix<IT, NT, OT>::operator=(CscMatrix<IT, NT, OT>&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, col_ptr, row_id, val);

    col_ptr = rhs.col_ptr;
    row_id = rhs.row_id;
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
CscMatrix<IT, NT, OT>::~CscMatrix()
{
  SafeDelete(memowned, col_ptr, row_id, val);
}

template <class IT, class NT, class OT>
void CscMatrix<IT, NT, OT>::Allocate(OT require_nnz, IT nRows, IT nCols)
{
  if (require_nnz <= 0) throw std::invalid_argument("CscMatrix::Allocate requires nnz > 0");
  SafeDelete(memowned, col_ptr, row_id, val);
  m = nRows;
  n = nCols;
  nnz = require_nnz;
  memowned = true;
  std::tie(col_ptr, row_id, val) = SafeAllocate(n, nnz);
}

template <class IT, class NT, class OT>
CscMatrix<IT, NT, OT> CscMatrix<IT, NT, OT>::Clone() const
{
  CscMatrix<IT, NT, OT> res;
  res.Allocate(nnz, m, n);
  if (col_ptr) {
    std::copy(col_ptr, col_ptr + n + 1, res.col_ptr);
  }
  if (row_id) {
    std::copy(row_id, row_id + nnz, res.row_id);
  }
  if (val) {
    std::copy(val, val + nnz, res.val);
  }
  return res;
}

template <class IT, class NT, class OT>
void CscMatrix<IT, NT, OT>::Reset()
{
  col_ptr = nullptr;
  row_id = nullptr;
  val = nullptr;
  nnz = 0;
  m = 0;
  n = 0;
  memowned = false;
}

}  // namespace spcraft
