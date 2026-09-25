#pragma once

#include <cstdint>
#include <vector>

namespace spcraft
{

using VI = std::vector<std::int32_t>;
using VVI = std::vector<std::vector<std::int32_t>>;
using VL = std::vector<std::int64_t>;
using VVL = std::vector<std::vector<std::int64_t>>;
using VF = std::vector<float>;
using VVF = std::vector<std::vector<float>>;
using VD = std::vector<double>;
using VVD = std::vector<std::vector<double>>;

#define VIT std::vector<IT>
#define VVIT std::vector<std::vector<IT>>
#define VNT std::vector<NT>
#define VVNT std::vector<std::vector<NT>>
#define VOT std::vector<OT>
#define VVOT std::vector<std::vector<OT>>

}  // namespace spcraft
