#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "core/AllocationGuard.h"

#include "utils/MatrixMacros.h"
#include "core/DenseVector.h"
#include "core/SparseVector.h"
#include "utils/omp/omp_wrapper.h"

namespace spcraft
{
/*******************************************************************************
 *                                Constructors                                 *
 ******************************************************************************/

SP_VEC_TEMP
SPSVEC::SparseVector() noexcept = default;

SP_VEC_TEMP
SPSVEC::SparseVector(IT count, IT length)
{
  std::tie(ind, val) = SafeAllocate(count, length);
  nnz = count;
  n = length;
  memowned = true;
}

SP_VEC_TEMP
SPSVEC::SparseVector(IT* ind_, NT* val_, IT count, IT length)
    : ind(ind_), val(val_), nnz(count), n(length), memowned(false)
{
}

SP_VEC_TEMP
SPSVEC::SparseVector(SPSVEC&& rhs) noexcept
    : ind(rhs.ind), val(rhs.val), nnz(rhs.nnz), n(rhs.n), memowned(rhs.memowned)
{
  rhs.Reset();
}

SP_VEC_TEMP
SPSVEC& SPSVEC::operator=(SPSVEC&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, ind, val);
    ind = rhs.ind;
    val = rhs.val;
    nnz = rhs.nnz;
    n = rhs.n;
    memowned = rhs.memowned;
    rhs.Reset();
  }
  return *this;
}

SP_VEC_TEMP
SPSVEC::~SparseVector() { SafeDelete(memowned, ind, val); }

/*******************************************************************************
 *                              Member Functions                               *
 ******************************************************************************/

SP_VEC_TEMP
void SPSVEC::Allocate(IT count, IT length)
{
  IT* new_ind = nullptr;
  NT* new_val = nullptr;
  std::tie(new_ind, new_val) = SafeAllocate(count, length);
  SafeDelete(memowned, ind, val);
  ind = new_ind;
  val = new_val;
  nnz = count;
  n = length;
  memowned = true;
}

SP_VEC_TEMP
SPSVEC SPSVEC::Clone() const
{
  SPSVEC ret(nnz, n);
  OMP_PARALLEL_FOR(schedule(static))
  for (IT k = 0; k < nnz; ++k) {
    ret.ind[k] = ind[k];
    ret.val[k] = val[k];
  }
  return ret;
}

SP_VEC_TEMP
void SPSVEC::Reset() noexcept
{
  ind = nullptr;
  val = nullptr;
  nnz = 0;
  n = 0;
  memowned = false;
}

SP_VEC_TEMP
SPDVEC SPSVEC::ToDense(NT fill) const
{
  // Each entry is checked against its predecessor, so the whole array is
  // strictly increasing exactly when no adjacent pair is out of order.
  int invalid_data = 0;
  OMP_PARALLEL_FOR(schedule(static) reduction(| : invalid_data))
  for (IT k = 0; k < nnz; ++k) {
    bool invalid = ind[k] >= n || (k > 0 && ind[k] <= ind[k - 1]);
    if constexpr (std::is_signed_v<IT>) invalid |= ind[k] < 0;
    invalid_data |= invalid;
  }
  if (invalid_data) {
    throw std::invalid_argument("SparseVector::ToDense indices must be strictly increasing within [0, n)");
  }

  SPDVEC dense(n, fill);
  // Indices are unique, so the scatter has no write conflicts.
  OMP_PARALLEL_FOR(schedule(static))
  for (IT k = 0; k < nnz; ++k) {
    dense.val[ind[k]] = val[k];
  }
  return dense;
}

/*******************************************************************************
 *                               Static Members                                *
 ******************************************************************************/

SP_VEC_TEMP
SPSVEC SPSVEC::FromDense(const SPDVEC& dense, NT fill)
{
  IT count = 0;
  OMP_PARALLEL_FOR(schedule(static) reduction(+ : count))
  for (IT i = 0; i < dense.n; ++i) {
    count += dense.val[i] != fill;
  }

  SPSVEC sparse(count, dense.n);
  // A serial compaction writes the entries in index order without a prefix scan.
  IT next = 0;
  for (IT i = 0; i < dense.n; ++i) {
    if (dense.val[i] != fill) {
      sparse.ind[next] = i;
      sparse.val[next] = dense.val[i];
      ++next;
    }
  }
  return sparse;
}

SP_VEC_TEMP
std::tuple<IT*, NT*> SPSVEC::SafeAllocate(IT count, IT length)
{
  if constexpr (std::is_signed_v<IT>) {
    if (count < 0 || length < 0) {
      throw std::invalid_argument("SparseVector nnz and length must be non-negative");
    }
  }
  if (count > length) {
    throw std::invalid_argument("SparseVector nnz cannot exceed its length");
  }

  // Calculate every size before allocating anything.
  const std::size_t index_count = detail::CheckedElementCount<IT>(count);
  const std::size_t value_count = detail::CheckedElementCount<NT>(count);
  if (count == IT{0}) return {nullptr, nullptr};

  IT* new_ind = static_cast<IT*>(std::malloc(index_count * sizeof(IT)));
  NT* new_val = static_cast<NT*>(std::malloc(value_count * sizeof(NT)));
  if (new_ind == nullptr || new_val == nullptr) {
    std::free(new_ind);
    std::free(new_val);
    throw std::bad_alloc();
  }
  return {new_ind, new_val};
}

SP_VEC_TEMP
void SPSVEC::SafeDelete(bool owned, IT* ind, NT* val) noexcept
{
  if (!owned) return;
  std::free(ind);
  std::free(val);
}

#define SPCRAFT_DEFINE_SPARSE_VECTOR(r, product) template class SparseVector<BOOST_PP_SEQ_ENUM(product)>;

// Retain Boolean graph storage, such as BFS frontiers, beside the numerical types.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_SPARSE_VECTOR,
                              ((std::int32_t)(std::int64_t))
                              ((float)(double)(bool)))
// clang-format on
#undef SPCRAFT_DEFINE_SPARSE_VECTOR

}  // namespace spcraft
