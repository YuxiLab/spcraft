#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace spcraft
{

template <class T>
using V = std::vector<T>;

template <class T>
using VV = V<V<T>>;

template <class T1, class T2>
using VPair = V<std::pair<T1, T2>>;

using VI = V<std::int32_t>;
using VVI = VV<std::int32_t>;
using VL = V<std::int64_t>;
using VVL = VV<std::int64_t>;
using VF = V<float>;
using VVF = VV<float>;
using VD = V<double>;
using VVD = VV<double>;

}  // namespace spcraft
