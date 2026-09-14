#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>

namespace spcraft
{
namespace detail
{

/**
 * @brief Convert an element count to std::size_t, refusing to overflow.
 *
 * `count * sizeof(T)` is computed in std::size_t by every allocation in the
 * core containers. When the product does not fit, the multiplication wraps and
 * the allocator is asked for a buffer far smaller than the caller believes it
 * received -- a silent heap overflow on the first write. Checking the count
 * against the largest allocatable number of T turns that into an exception.
 *
 * @param count Requested number of elements; must already be non-negative.
 * @return The same count as std::size_t.
 * @throws std::bad_array_new_length when count elements of T cannot be sized.
 */
template <class T, class CountType>
[[nodiscard]] inline std::size_t CheckedElementCount(CountType count)
{
  static_assert(std::is_integral_v<CountType>, "element counts must be integral");
  if (static_cast<std::uintmax_t>(count) >
      static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
    throw std::bad_array_new_length();
  }
  return static_cast<std::size_t>(count);
}

}  // namespace detail
}  // namespace spcraft
