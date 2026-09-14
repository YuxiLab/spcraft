#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>

#include "core/AllocationGuard.h"
#include "core/DenseVector.h"
#include "utils/omp/omp_wrapper.h"

namespace spcraft
{
/*************************************
 *             constructor
 *************************************/

template <class IT, class NT>
DenseVector<IT, NT>::DenseVector(IT size)
{
  val = SafeAllocate(size);
  n = size;
  memowned = true;
}

template <class IT, class NT>
DenseVector<IT, NT>::DenseVector(IT size, NT initval)
{
  val = SafeAllocate(size);
  n = size;
  memowned = true;
  OMP_PARALLEL_FOR(schedule(static, 32))
  for (size_t i = 0; i < size; ++i) {
    val[i] = initval;
  }
}

template <class IT, class NT>
DenseVector<IT, NT>::DenseVector(DenseVector<IT, NT>&& rhs) noexcept
    : val(rhs.val), n(rhs.n), memowned(rhs.memowned)
{
  rhs.Reset();
}

template <class IT, class NT>
DenseVector<IT, NT>& DenseVector<IT, NT>::operator=(DenseVector<IT, NT>&& rhs) noexcept
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

template <class IT, class NT>
DenseVector<IT, NT>::~DenseVector()
{
  SafeDelete(memowned, val);
}
/*************************************
 *             member functions
 *************************************/

template <class IT, class NT>
void DenseVector<IT, NT>::Allocate(IT size)
{
  NT* replacement = SafeAllocate(size);
  SafeDelete(memowned, val);
  val = replacement;
  n = size;
  memowned = true;
}

//! return a deep copy instance with owned memory.
template <class IT, class NT>
DenseVector<IT, NT> DenseVector<IT, NT>::Clone() const
{
  DenseVector<IT, NT> ret(n);  //! own-memoried inst with uninit memory.
  OMP_PARALLEL_FOR()
  for (IT i = 0; i < n; ++i) {
    ret.val[i] = val[i];
  }
  return ret;
}

template <class IT, class NT>
void DenseVector<IT, NT>::Reset() noexcept
{
  val = nullptr;
  n = 0;
  memowned = false;
}

template <class IT, class NT>
void DenseVector<IT, NT>::Random(int seed)
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

/*************************************
 *             Static Members
 *************************************/
template <class IT, class NT>
NT* DenseVector<IT, NT>::SafeAllocate(IT size)
{
  // size should be >= 0
  if constexpr (std::is_signed_v<IT>) {
    if (size < 0) {
      throw std::invalid_argument("DenseVector size must be non-negative");
    }
  }
  if (size == IT{0}) return nullptr;
  // protect size overflow
  auto tmp = detail::CheckedElementCount<NT>(size);
  NT* ptr = static_cast<NT*>(malloc(tmp * sizeof(NT)));
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

template <class IT, class NT>
void DenseVector<IT, NT>::SafeDelete(bool owned, NT* ptr) noexcept
{
  if (owned && ptr != nullptr) free(ptr);
}

}  // namespace spcraft
