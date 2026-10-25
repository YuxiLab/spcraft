#include "core/DenseMatrix.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "core/AllocationGuard.h"

namespace spcraft
{

template <class IT, class NT, class OT>
DenseMatrix<IT, NT, OT>::DenseMatrix() noexcept = default;

template <class IT, class NT, class OT>
DenseMatrix<IT, NT, OT>::DenseMatrix(IT rows, IT cols)
{
  // 1. Allocate packed column-major storage without initializing its values.
  Allocate(rows, cols);
}

template <class IT, class NT, class OT>
DenseMatrix<IT, NT, OT>::DenseMatrix(IT rows, IT cols, NT value)
{
  // 1. Allocate a packed owner, then initialize only its actual entries.
  Allocate(rows, cols);
  if (size() != OT{0}) {
    std::fill_n(val, static_cast<std::size_t>(size()), value);
  }
}

template <class IT, class NT, class OT>
DenseMatrix<IT, NT, OT>::DenseMatrix(NT* values, IT rows, IT cols)
    : DenseMatrix(values, rows, cols, static_cast<OT>(rows))
{
}

template <class IT, class NT, class OT>
DenseMatrix<IT, NT, OT>::DenseMatrix(NT* values, IT rows, IT cols, OT leading_dim)
{
  // 1. Validate the layout before accepting a view; a nonempty view needs data.
  const OT span = CheckedSpan(rows, cols, leading_dim);
  if (span != OT{0} && values == nullptr) {
    throw std::invalid_argument("Nonempty dense matrix view requires storage");
  }
  // 2. Borrow the caller's buffer and stride; its lifetime remains the caller's.
  val = values;
  m = rows;
  n = cols;
  ld = leading_dim;
  memowned = false;
}

template <class IT, class NT, class OT>
DenseMatrix<IT, NT, OT>::DenseMatrix(DenseMatrix&& other) noexcept
{
  // 1. Transfer ownership through the same path used by move assignment.
  *this = std::move(other);
}

template <class IT, class NT, class OT>
DenseMatrix<IT, NT, OT>& DenseMatrix<IT, NT, OT>::operator=(DenseMatrix&& other) noexcept
{
  // 1. Preserve self-moves; otherwise release only this object's owned buffer.
  if (this != &other) {
    Clear();
    // 2. Transfer the pointer, shape, stride, and ownership flag together.
    val = other.val;
    m = other.m;
    n = other.n;
    ld = other.ld;
    memowned = other.memowned;
    // 3. Empty the source without freeing the transferred storage.
    other.val = nullptr;
    other.m = IT{0};
    other.n = IT{0};
    other.ld = OT{0};
    other.memowned = true;
  }
  return *this;
}

template <class IT, class NT, class OT>
DenseMatrix<IT, NT, OT>::~DenseMatrix()
{
  // 1. Views detach; owners release their allocation.
  Clear();
}

template <class IT, class NT, class OT>
void DenseMatrix<IT, NT, OT>::Allocate(IT rows, IT cols)
{
  // 1. Validate the packed shape and bytes before multiplication or allocation.
  const OT leading_dim = static_cast<OT>(rows);
  const OT count = CheckedSpan(rows, cols, leading_dim);
  NT* replacement = nullptr;
  if (count != OT{0}) {
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(NT);
    replacement = static_cast<NT*>(std::malloc(bytes));
    if (replacement == nullptr) {
      throw std::bad_alloc();
    }
  }
  // 2. Commit only after success so a rejected size or allocation preserves state.
  Clear();
  val = replacement;
  m = rows;
  n = cols;
  ld = leading_dim;
  memowned = true;
}

template <class IT, class NT, class OT>
DenseMatrix<IT, NT, OT> DenseMatrix<IT, NT, OT>::Clone() const
{
  // 1. Allocate independent packed storage, preserving even a zero-area shape.
  DenseMatrix result(m, n);
  // 2. Copy logical entries using each matrix's stride, skipping view padding.
  if (m != IT{0} && n != IT{0}) {
    for (IT col = 0; col < n; ++col) {
      for (IT row = 0; row < m; ++row) {
        result(row, col) = (*this)(row, col);
      }
    }
  }
  return result;
}

template <class IT, class NT, class OT>
void DenseMatrix<IT, NT, OT>::Clear() noexcept
{
  // 1. Free only owned memory; an external view's storage remains untouched.
  if (memowned) {
    std::free(val);
  }
  // 2. Restore a reusable empty owner with no stale dimensions or stride.
  val = nullptr;
  m = IT{0};
  n = IT{0};
  ld = OT{0};
  memowned = true;
}

template <class IT, class NT, class OT>
OT DenseMatrix<IT, NT, OT>::CheckedSpan(IT rows, IT cols, OT leading_dim)
{
  static_assert(std::is_integral_v<IT> && std::is_signed_v<IT>);
  static_assert(std::is_integral_v<OT> && std::is_signed_v<OT>);
  static_assert(std::numeric_limits<OT>::digits >= std::numeric_limits<IT>::digits);
  // 1. Reject invalid dimensions/strides before converting coordinates to offsets.
  if (rows < IT{0} || cols < IT{0} || leading_dim < static_cast<OT>(rows)) {
    throw std::invalid_argument("Dense matrix requires nonnegative dimensions and ld >= rows");
  }
  if (rows == IT{0} || cols == IT{0}) {
    return OT{0};
  }
  // 2. Check the last column's start plus its rows without overflowing OT.
  const OT row_count = static_cast<OT>(rows);
  const OT last_col = static_cast<OT>(cols) - OT{1};
  if (last_col > (std::numeric_limits<OT>::max() - row_count) / leading_dim) {
    throw std::bad_array_new_length();
  }
  const OT span = last_col * leading_dim + row_count;
  // 3. Check the complete accessible span in bytes, including gaps between columns.
  (void)detail::CheckedElementCount<NT>(span);
  return span;
}

#define SPCRAFT_DEFINE_DENSE_MATRIX(r, product)                                   \
  template class DenseMatrix<BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                             BOOST_PP_SEQ_ELEM(1, product),                       \
                             BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product))>;

// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_DENSE_MATRIX,
                              (((int32_t)(int32_t))
                               ((int32_t)(int64_t))
                               ((int64_t)(int64_t)))
                              ((float)(double)(bool)))
// clang-format on
#undef SPCRAFT_DEFINE_DENSE_MATRIX

}  // namespace spcraft
