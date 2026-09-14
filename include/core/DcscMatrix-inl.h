#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/AllocationGuard.h"
#include "core/DcscMatrix.h"

namespace spcraft
{
/*************************************
 *             constructor
 *************************************/

template <class IT, class NT, class OT>
DcscMatrix<IT, NT, OT>& DcscMatrix<IT, NT, OT>::operator=(DcscMatrix&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, col_ptr, col_id, row_id, val);
    col_ptr = rhs.col_ptr;
    col_id = rhs.col_id;
    row_id = rhs.row_id;
    val = rhs.val;
    nnz = rhs.nnz;
    m = rhs.m;
    n = rhs.n;
    nzc = rhs.nzc;
    memowned = rhs.memowned;
    rhs.Reset();
  }
  return *this;
}

/*************************************
 *             member functions
 *************************************/

template <class IT, class NT, class OT>
void DcscMatrix<IT, NT, OT>::Allocate(OT nonzeros, IT nRows, IT nCols, IT storedColumns)
{
  static_assert(std::is_integral_v<IT> && std::is_integral_v<OT>);
  if constexpr (std::is_signed_v<IT>) {
    if (nRows < 0 || nCols < 0 || storedColumns < 0) {
      throw std::invalid_argument("DcscMatrix dimensions/counts must be non-negative");
    }
  }
  if constexpr (std::is_signed_v<OT>) {
    if (nonzeros < 0) throw std::invalid_argument("DcscMatrix nnz must be non-negative");
  }
  if (storedColumns > nCols ||
      static_cast<std::uintmax_t>(storedColumns) > static_cast<std::uintmax_t>(nonzeros) ||
      ((nonzeros == 0) != (storedColumns == 0)) || (nonzeros != 0 && nRows == 0)) {
    throw std::invalid_argument("DcscMatrix inconsistent nonempty-column count");
  }
  const auto replacement = SafeAllocate(storedColumns, nonzeros);
  SafeDelete(memowned, col_ptr, col_id, row_id, val);
  std::tie(col_ptr, col_id, row_id, val) = replacement;
  nnz = nonzeros;
  m = nRows;
  n = nCols;
  nzc = storedColumns;
  memowned = true;
}

template <class IT, class NT, class OT>
DcscMatrix<IT, NT, OT> DcscMatrix<IT, NT, OT>::Clone() const
{
  DcscMatrix result;
  result.Allocate(nnz, m, n, nzc);
  if (col_ptr) std::copy_n(col_ptr, static_cast<std::size_t>(nzc) + 1, result.col_ptr);
  if (nzc != 0) std::copy_n(col_id, nzc, result.col_id);
  if (nnz != 0) {
    std::copy_n(row_id, nnz, result.row_id);
    std::copy_n(val, nnz, result.val);
  }
  return result;
}

template <class IT, class NT, class OT>
void DcscMatrix<IT, NT, OT>::Reset()
{
  col_ptr = nullptr;
  col_id = nullptr;
  row_id = nullptr;
  val = nullptr;
  nnz = 0;
  m = n = nzc = 0;
  memowned = false;
}

template <class IT, class NT, class OT>
DcscMatrix<IT, NT, OT> DcscMatrix<IT, NT, OT>::FromCsc(const CscMatrix<IT, NT, OT>& source)
{
  IT columns = 0;
  for (IT col = 0; col < source.n; ++col) {
    if (source.col_ptr[col] != source.col_ptr[col + 1]) ++columns;
  }
  DcscMatrix result;
  result.Allocate(source.nnz, source.m, source.n, columns);
  IT slot = 0;
  for (IT col = 0; col < source.n; ++col) {
    if (source.col_ptr[col] != source.col_ptr[col + 1]) {
      result.col_id[slot] = col;
      result.col_ptr[slot++] = source.col_ptr[col];
    }
  }
  result.col_ptr[columns] = source.nnz;
  if (source.nnz != 0) {
    std::copy_n(source.row_id, source.nnz, result.row_id);
    std::copy_n(source.val, source.nnz, result.val);
  }
  return result;
}

template <class IT, class NT, class OT>
CscMatrix<IT, NT, OT> DcscMatrix<IT, NT, OT>::ToCsc() const
{
  CscMatrix<IT, NT, OT> result;
  result.Allocate(nnz, m, n);
  IT slot = 0;
  OT offset = 0;
  for (IT col = 0; col < n; ++col) {
    result.col_ptr[col] = offset;
    if (slot < nzc && col_id[slot] == col) offset = col_ptr[++slot];
  }
  result.col_ptr[n] = nnz;
  if (nnz != 0) {
    std::copy_n(row_id, nnz, result.row_id);
    std::copy_n(val, nnz, result.val);
  }
  return result;
}

/*************************************
 *             Static Members
 *************************************/

template <class IT, class NT, class OT>
std::tuple<OT*, IT*, IT*, NT*> DcscMatrix<IT, NT, OT>::SafeAllocate(IT nzc, OT nnz)
{
  const auto columns = detail::CheckedElementCount<IT>(nzc);
  const auto offsets = detail::CheckedElementCount<OT>(nzc);
  if (offsets == std::numeric_limits<std::size_t>::max()) {
    throw std::bad_array_new_length();
  }
  const auto pointers = detail::CheckedElementCount<OT>(offsets + 1);
  const auto rows = detail::CheckedElementCount<IT>(nnz);
  const auto values = detail::CheckedElementCount<NT>(nnz);
  auto* p = static_cast<OT*>(std::calloc(pointers, sizeof(OT)));
  auto* c = columns ? static_cast<IT*>(std::calloc(columns, sizeof(IT))) : nullptr;
  auto* r = rows ? static_cast<IT*>(std::calloc(rows, sizeof(IT))) : nullptr;
  auto* v = values ? static_cast<NT*>(std::calloc(values, sizeof(NT))) : nullptr;
  if (!p || (columns && !c) || (rows && !r) || (values && !v)) {
    SafeDelete(true, p, c, r, v);
    throw std::bad_alloc();
  }
  return {p, c, r, v};
}

template <class IT, class NT, class OT>
void DcscMatrix<IT, NT, OT>::SafeDelete(bool owned, OT* pointers, IT* columns, IT* rows, NT* values)
{
  if (owned) {
    std::free(pointers);
    std::free(columns);
    std::free(rows);
    std::free(values);
  }
}
}  // namespace spcraft
