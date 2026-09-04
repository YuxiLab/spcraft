#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>

#include "core/DenseVector.h"

namespace spcraft
{

template <class IT, class NT>
NT* DenseVector<IT, NT>::SafeAllocate(IT size)
{
  if (size < IT{0}) {
    throw std::invalid_argument("DenseVector size must be non-negative");
  }
  if (size == IT{0}) return nullptr;

  if (static_cast<std::uintmax_t>(size) >
      static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max() / sizeof(NT))) {
    throw std::bad_array_new_length();
  }

  NT* result = static_cast<NT*>(std::malloc(static_cast<std::size_t>(size) * sizeof(NT)));
  if (result == nullptr) throw std::bad_alloc();
  return result;
}

template <class IT, class NT>
void DenseVector<IT, NT>::SafeDelete(bool memowned, NT* val)
{
  if (memowned) std::free(val);
}

template <class IT, class NT>
DenseVector<IT, NT>::DenseVector(IT size)
{
  Allocate(size);
}

template <class IT, class NT>
DenseVector<IT, NT>::DenseVector(NT* val_, IT size) : val(val_), n(size), memowned(false)
{
  if (size < IT{0}) {
    throw std::invalid_argument("DenseVector size must be non-negative");
  }
  if (size != IT{0} && val == nullptr) {
    throw std::invalid_argument("DenseVector data must not be null for a non-empty vector");
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

template <class IT, class NT>
void DenseVector<IT, NT>::Allocate(IT size)
{
  if (size < IT{0}) {
    throw std::invalid_argument("DenseVector size must be non-negative");
  }
  NT* replacement = SafeAllocate(size);
  SafeDelete(memowned, val);
  val = replacement;
  n = size;
  memowned = true;
}

template <class IT, class NT>
DenseVector<IT, NT> DenseVector<IT, NT>::Clone() const
{
  DenseVector<IT, NT> result(n);
  if (n != IT{0}) {
    std::copy(val, val + n, result.val);
  }
  return result;
}

template <class IT, class NT>
void DenseVector<IT, NT>::Reset() noexcept
{
  val = nullptr;
  n = 0;
  memowned = false;
}

}  // namespace spcraft
