#pragma once

#include <cstddef>

namespace spcraft::examples
{

template <class Index, class Number, std::size_t TileSize>
void DescribeKernelVariant();

void DescribeAllKernelVariants();

}  // namespace spcraft::examples
