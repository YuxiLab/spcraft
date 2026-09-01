#pragma once

#include <algorithm>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "CooMatrix.h"

namespace spcraft
{

template <class IT, class NT, class OT>
std::tuple<IT*, IT*, NT*> CooMatrix<IT, NT, OT>::SafeAllocate(OT nnz)
{
  IT* r = (nnz > 0) ? new IT[nnz]() : nullptr;
  IT* c = (nnz > 0) ? new IT[nnz]() : nullptr;
  NT* v = (nnz > 0) ? new NT[nnz]() : nullptr;
  return std::make_tuple(r, c, v);
}

template <class IT, class NT, class OT>
void CooMatrix<IT, NT, OT>::SafeDelete(bool memowned, IT* row_id, IT* col_id, NT* val)
{
  if (memowned) {
    delete[] row_id;
    delete[] col_id;
    delete[] val;
  }
}

template <class IT, class NT, class OT>
CooMatrix<IT, NT, OT>::CooMatrix(IT* row_id_, IT* col_id_, NT* val_, OT nnz_, IT m_, IT n_)
    : row_id(row_id_), col_id(col_id_), val(val_), nnz(nnz_), m(m_), n(n_), memowned(false)
{
}

template <class IT, class NT, class OT>
CooMatrix<IT, NT, OT>::CooMatrix(CooMatrix<IT, NT, OT>&& rhs) noexcept
    : row_id(rhs.row_id),
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
CooMatrix<IT, NT, OT>& CooMatrix<IT, NT, OT>::operator=(CooMatrix<IT, NT, OT>&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, row_id, col_id, val);

    row_id = rhs.row_id;
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
CooMatrix<IT, NT, OT>::~CooMatrix()
{
  SafeDelete(memowned, row_id, col_id, val);
}

template <class IT, class NT, class OT>
void CooMatrix<IT, NT, OT>::Allocate(OT require_nnz, IT nRows, IT nCols)
{
  SafeDelete(memowned, row_id, col_id, val);
  m = nRows;
  n = nCols;
  nnz = require_nnz;
  memowned = true;
  std::tie(row_id, col_id, val) = SafeAllocate(nnz);
}

template <class IT, class NT, class OT>
CooMatrix<IT, NT, OT> CooMatrix<IT, NT, OT>::Clone() const
{
  CooMatrix<IT, NT, OT> res;
  res.Allocate(nnz, m, n);
  if (row_id) {
    std::copy(row_id, row_id + nnz, res.row_id);
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
void CooMatrix<IT, NT, OT>::Reset()
{
  row_id = nullptr;
  col_id = nullptr;
  val = nullptr;
  nnz = 0;
  m = 0;
  n = 0;
  memowned = false;
}

template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT> CooMatrix<IT, NT, OT>::ToCsr() const
{
  CsrMatrix<IT, NT, OT> csr;
  csr.Allocate(nnz, m, n);

  if (m == 0) return csr;

  std::fill(csr.row_ptr, csr.row_ptr + m + 1, 0);

  for (OT i = 0; i < nnz; ++i) {
    IT r = row_id[i];
    if (r >= 0 && r < m) {
      csr.row_ptr[r + 1]++;
    }
  }

  for (IT i = 0; i < m; ++i) {
    csr.row_ptr[i + 1] += csr.row_ptr[i];
  }

  std::vector<OT> offsets(csr.row_ptr, csr.row_ptr + m);
  for (OT i = 0; i < nnz; ++i) {
    IT r = row_id[i];
    if (r >= 0 && r < m) {
      OT dest = offsets[r]++;
      csr.col_id[dest] = col_id[i];
      csr.val[dest] = val[i];
    }
  }

  return csr;
}

template <class IT, class NT, class OT>
CooMatrix<IT, NT, OT> CooMatrix<IT, NT, OT>::FromMatrixMarket(const std::string& filename)
{
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    throw std::runtime_error("CooMatrix::FromMatrixMarket failed: unable to open file " +
                             filename);
  }

  int64_t rows = 0;
  int64_t cols = 0;
  std::vector<IT> r_vec;
  std::vector<IT> c_vec;
  std::vector<NT> v_vec;

  fast_matrix_market::read_matrix_market_triplet(ifs, rows, cols, r_vec, c_vec, v_vec);

  CooMatrix<IT, NT, OT> coo;
  OT num_entries = static_cast<OT>(r_vec.size());
  coo.Allocate(num_entries, static_cast<IT>(rows), static_cast<IT>(cols));

  if (num_entries > 0) {
    std::copy(r_vec.begin(), r_vec.end(), coo.row_id);
    std::copy(c_vec.begin(), c_vec.end(), coo.col_id);
    std::copy(v_vec.begin(), v_vec.end(), coo.val);
  }

  return coo;
}

}  // namespace spcraft
