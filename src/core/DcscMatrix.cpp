#include <algorithm>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/AllocationGuard.h"
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "utils/MatrixMacros.h"
#include "core/DcscMatrix.h"
#include "utils/omp/omp_wrapper.h"

namespace spcraft
{
/*******************************************************************************
 *                                Constructors                                 *
 ******************************************************************************/

SP_MAT_TEMP
SPDCSC::DcscMatrix() noexcept = default;

SP_MAT_TEMP
SPDCSC::DcscMatrix(OT* pointers, IT* columns, IT* rows, NT* values, OT nonzeros, IT nRows, IT nCols,
                   IT storedColumns)
    : col_ptr(pointers),
      col_id(columns),
      row_id(rows),
      val(values),
      nnz(nonzeros),
      m(nRows),
      n(nCols),
      nzc(storedColumns),
      memowned(false)
{
}

SP_MAT_TEMP
SPDCSC::DcscMatrix(DcscMatrix&& rhs) noexcept
    : col_ptr(rhs.col_ptr),
      col_id(rhs.col_id),
      row_id(rhs.row_id),
      val(rhs.val),
      nnz(rhs.nnz),
      m(rhs.m),
      n(rhs.n),
      nzc(rhs.nzc),
      memowned(rhs.memowned)
{
  rhs.Reset();
}

SP_MAT_TEMP
SPDCSC::~DcscMatrix() { SafeDelete(memowned, col_ptr, col_id, row_id, val); }

SP_MAT_TEMP
SPDCSC& SPDCSC::operator=(DcscMatrix&& rhs) noexcept
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

/*******************************************************************************
 *                              Member Functions                               *
 ******************************************************************************/

SP_MAT_TEMP
void SPDCSC::Allocate(OT nonzeros, IT nRows, IT nCols, IT storedColumns)
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
  const std::tuple<OT*, IT*, IT*, NT*> replacement = SafeAllocate(storedColumns, nonzeros);
  SafeDelete(memowned, col_ptr, col_id, row_id, val);
  std::tie(col_ptr, col_id, row_id, val) = replacement;
  nnz = nonzeros;
  m = nRows;
  n = nCols;
  nzc = storedColumns;
  memowned = true;
}

SP_MAT_TEMP
SPDCSC SPDCSC::Clone() const
{
  SPDCSC result;
  result.Allocate(nnz, m, n, nzc);
  if (col_ptr) std::copy_n(col_ptr, static_cast<std::size_t>(nzc) + 1, result.col_ptr);
  if (nzc != 0) std::copy_n(col_id, nzc, result.col_id);
  if (nnz != 0) {
    std::copy_n(row_id, nnz, result.row_id);
    std::copy_n(val, nnz, result.val);
  }
  return result;
}

SP_MAT_TEMP
void SPDCSC::Reset()
{
  col_ptr = nullptr;
  col_id = nullptr;
  row_id = nullptr;
  val = nullptr;
  nnz = 0;
  m = n = nzc = 0;
  memowned = false;
}

SP_MAT_TEMP
SPDCSC SPDCSC::FromCsc(const SPCSC& source)
{
  IT columns = 0;
  for (IT col = 0; col < source.n; ++col) {
    if (source.col_ptr[col] != source.col_ptr[col + 1]) ++columns;
  }
  SPDCSC result;
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

SP_MAT_TEMP
SPCSC SPDCSC::ToCsc() const
{
  SPCSC result;
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

/*******************************************************************************
 *                               Static Members                                *
 ******************************************************************************/

SP_MAT_TEMP
std::tuple<OT*, IT*, IT*, NT*> SPDCSC::SafeAllocate(IT nzc, OT nnz)
{
  const std::size_t columns = detail::CheckedElementCount<IT>(nzc);
  const std::size_t offsets = detail::CheckedElementCount<OT>(nzc);
  if (offsets == std::numeric_limits<std::size_t>::max()) {
    throw std::bad_array_new_length();
  }
  const std::size_t pointers = detail::CheckedElementCount<OT>(offsets + 1);
  const std::size_t rows = detail::CheckedElementCount<IT>(nnz);
  const std::size_t values = detail::CheckedElementCount<NT>(nnz);
  OT* p = static_cast<OT*>(std::calloc(pointers, sizeof(OT)));
  IT* c = columns ? static_cast<IT*>(std::calloc(columns, sizeof(IT))) : nullptr;
  IT* r = rows ? static_cast<IT*>(std::calloc(rows, sizeof(IT))) : nullptr;
  NT* v = values ? static_cast<NT*>(std::calloc(values, sizeof(NT))) : nullptr;
  if (!p || (columns && !c) || (rows && !r) || (values && !v)) {
    SafeDelete(true, p, c, r, v);
    throw std::bad_alloc();
  }
  return {p, c, r, v};
}

SP_MAT_TEMP
void SPDCSC::SafeDelete(bool owned, OT* pointers, IT* columns, IT* rows, NT* values)
{
  if (owned) {
    std::free(pointers);
    std::free(columns);
    std::free(rows);
    std::free(values);
  }
}

#define SPCRAFT_DEFINE_DCSC_MATRIX(r, product)                                   \
  template class DcscMatrix<BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                            BOOST_PP_SEQ_ELEM(1, product),                       \
                            BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product))>;

// Match the compiled kernels' storage pairs and retain Boolean graph storage.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_DCSC_MATRIX,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)(bool)))
// clang-format on
#undef SPCRAFT_DEFINE_DCSC_MATRIX

}  // namespace spcraft
