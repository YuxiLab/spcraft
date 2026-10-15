#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "core/AllocationGuard.h"

#include "utils/MatrixMacros.h"
#include "core/DenseVector.h"
#include "utils/omp/omp_wrapper.h"

namespace spcraft
{
/*******************************************************************************
 *                                Constructors                                 *
 ******************************************************************************/

SP_VEC_TEMP
SPDVEC::DenseVector() noexcept = default;

SP_VEC_TEMP
SPDVEC::DenseVector(NT* val_, IT size) : val(val_), n(size), memowned(false) {}

SP_VEC_TEMP
SPDVEC::DenseVector(IT size)
{
  val = SafeAllocate(size);
  n = size;
  memowned = true;
}

SP_VEC_TEMP
SPDVEC::DenseVector(IT size, NT initval)
{
  val = SafeAllocate(size);
  n = size;
  memowned = true;
  OMP_PARALLEL_FOR(schedule(static))
  for (IT i = 0; i < size; ++i) {
    val[i] = initval;
  }
}

SP_VEC_TEMP
SPDVEC::DenseVector(SPDVEC&& rhs) noexcept : val(rhs.val), n(rhs.n), memowned(rhs.memowned) { rhs.Reset(); }

SP_VEC_TEMP
SPDVEC& SPDVEC::operator=(SPDVEC&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, val);
    val = rhs.val;
    n = rhs.n;
    memowned = rhs.memowned;
    rhs.Reset();
  }
  return *this;
}

SP_VEC_TEMP
SPDVEC::~DenseVector() { SafeDelete(memowned, val); }
/*******************************************************************************
 *                              Member Functions                               *
 ******************************************************************************/

SP_VEC_TEMP
void SPDVEC::Allocate(IT size)
{
  NT* replacement = SafeAllocate(size);
  SafeDelete(memowned, val);
  val = replacement;
  n = size;
  memowned = true;
}

//! return a deep copy instance with owned memory.
SP_VEC_TEMP
SPDVEC SPDVEC::Clone() const
{
  SPDVEC ret(n);  //! own-memoried inst with uninit memory.
  OMP_PARALLEL_FOR()
  for (IT i = 0; i < n; ++i) {
    ret.val[i] = val[i];
  }
  return ret;
}

SP_VEC_TEMP
void SPDVEC::Reset() noexcept
{
  val = nullptr;
  n = 0;
  memowned = false;
}

SP_VEC_TEMP
void SPDVEC::Random(int seed)
{
  static_assert(std::is_arithmetic_v<NT>, "DenseVector::Random requires an arithmetic value type");
  std::mt19937_64 rng(static_cast<std::mt19937_64::result_type>(seed));
  if constexpr (std::is_floating_point_v<NT>) {
    std::uniform_real_distribution<NT> distribution(NT{-1}, NT{1});
    for (IT i = 0; i < n; ++i) {
      val[i] = distribution(rng);
    }
  } else {
    using RandomInt = std::conditional_t<std::is_signed_v<NT>, std::intmax_t, std::uintmax_t>;
    std::uniform_int_distribution<RandomInt> distribution(std::is_signed_v<NT> ? -1 : 0, 1);
    for (IT i = 0; i < n; ++i) {
      val[i] = static_cast<NT>(distribution(rng));
    }
  }
}

/*******************************************************************************
 *                               Static Members                                *
 ******************************************************************************/
SP_VEC_TEMP
NT* SPDVEC::SafeAllocate(IT size)
{
  // size should be >= 0
  if constexpr (std::is_signed_v<IT>) {
    if (size < 0) {
      throw std::invalid_argument("DenseVector size must be non-negative");
    }
  }
  if (size == IT{0}) return nullptr;
  // protect size overflow
  const std::size_t count = detail::CheckedElementCount<NT>(size);
  NT* ptr = static_cast<NT*>(std::malloc(count * sizeof(NT)));
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

SP_VEC_TEMP
void SPDVEC::SafeDelete(bool owned, NT* ptr) noexcept
{
  if (owned && ptr != nullptr) std::free(ptr);
}

#define SPCRAFT_DEFINE_DENSE_VECTOR(r, product) template class DenseVector<BOOST_PP_SEQ_ENUM(product)>;

// Retain Boolean graph storage alongside the floating-point numerical types.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_DENSE_VECTOR,
                              ((std::int32_t)(std::int64_t))
                              ((float)(double)(bool)))
// clang-format on
#undef SPCRAFT_DEFINE_DENSE_VECTOR

}  // namespace spcraft
