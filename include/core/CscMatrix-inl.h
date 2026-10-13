#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/AllocationGuard.h"
#include "core/CscMatrix.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{
/*************************************
 *             constructor
 *************************************/
SP_MAT_TEMP
SPCSC& SPCSC::operator=(SPCSC&& rhs) noexcept
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

SP_MAT_TEMP
SPCSC::~CscMatrix() { SafeDelete(memowned, col_ptr, row_id, val); }

SP_MAT_TEMP
void SPCSC::Allocate(OT require_nnz, IT nRows, IT nCols)
{
  if constexpr (std::is_signed_v<IT>) {
    if (nRows < 0 || nCols < 0) {
      throw std::invalid_argument("CscMatrix dimensions must be non-negative");
    }
  }
  if constexpr (std::is_signed_v<OT>) {
    if (require_nnz < 0) {
      throw std::invalid_argument("CscMatrix nnz must be non-negative");
    }
  }
  // Build the replacement first: a throwing allocation must leave the matrix
  // exactly as it was, still owning storage it can free.
  auto replacement = SafeAllocate(nCols, require_nnz);
  SafeDelete(memowned, col_ptr, row_id, val);
  m = nRows;
  n = nCols;
  nnz = require_nnz;
  memowned = true;
  std::tie(col_ptr, row_id, val) = replacement;
}

/*************************************
 *             member functions
 *************************************/
SP_MAT_TEMP
SPCSC SPCSC::Clone() const
{
  SPCSC res;
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

SP_MAT_TEMP
void SPCSC::Reset()
{
  col_ptr = nullptr;
  row_id = nullptr;
  val = nullptr;
  nnz = 0;
  m = 0;
  n = 0;
  memowned = false;
}

/*************************************
 *             Static Members
 *************************************/

SP_MAT_TEMP
std::tuple<OT*, IT*, NT*> SPCSC::SafeAllocate(IT n, OT nnz)
{
  if constexpr (std::is_signed_v<IT>) {
    if (n < 0) throw std::invalid_argument("CscMatrix column count must be non-negative");
  }
  if constexpr (std::is_signed_v<OT>) {
    if (nnz < 0) throw std::invalid_argument("CscMatrix nnz must be non-negative");
  }

  // n + 1 is computed in std::size_t; in IT it would overflow at the maximum
  // representable column count.
  // Size every array before allocating any of them, so a request that cannot be
  // sized throws while there is still nothing to free.
  const std::size_t pointer_count = detail::CheckedElementCount<OT>(n) + 1;
  const std::size_t index_count =
      nnz != OT{0} ? detail::CheckedElementCount<IT>(nnz) : std::size_t{0};
  const std::size_t value_count =
      nnz != OT{0} ? detail::CheckedElementCount<NT>(nnz) : std::size_t{0};

  OT* c = static_cast<OT*>(std::calloc(pointer_count, sizeof(OT)));
  IT* r = nullptr;
  NT* v = nullptr;
  if (nnz != OT{0}) {
    r = static_cast<IT*>(std::calloc(index_count, sizeof(IT)));
    v = static_cast<NT*>(std::calloc(value_count, sizeof(NT)));
  }
  if (c == nullptr || (nnz != OT{0} && (r == nullptr || v == nullptr))) {
    std::free(c);
    std::free(r);
    std::free(v);
    throw std::bad_alloc();
  }
  return std::make_tuple(c, r, v);
}

SP_MAT_TEMP
void SPCSC::SafeDelete(bool memowned, OT* col_ptr, IT* row_id, NT* val)
{
  // The pointers are by value, so nulling them here would be a dead store.
  // Callers are responsible for clearing or overwriting their own members.
  if (memowned) {
    std::free(col_ptr);
    std::free(row_id);
    std::free(val);
  }
}

}  // namespace spcraft
