#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/AllocationGuard.h"

namespace spcraft
{
namespace detail
{

template <class Entry>
[[nodiscard]] inline std::size_t SpGEMMHashCapacity(std::uintmax_t count)
{
  if (count == 0) {
    return 0;
  }
  constexpr std::size_t kMinCapacity = 16;
  std::size_t capacity = kMinCapacity;
  while (capacity < count) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("SpGEMM hash capacity exceeds size_t");
    }
    capacity *= 2;
  }
  const auto size = CheckedElementCount<Entry>(capacity);
  if (size > std::vector<Entry>().max_size()) {
    throw std::length_error("SpGEMM hash capacity exceeds vector max_size");
  }
  return size;
}

template <class IT>
[[nodiscard]] inline std::size_t SpGEMMHashSlot(IT column, std::size_t mask)
{
  // Match CombBLAS's multiplicative hash, with unsigned arithmetic so large
  // column indices cannot trigger signed multiplication overflow.
  constexpr std::size_t kHashScale = 107;
  return (static_cast<std::size_t>(column) * kHashScale) & mask;
}

}  // namespace detail

}  // namespace spcraft
