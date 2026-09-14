#include "e01_multi_template_instantiation.h"

#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>

namespace spcraft::examples
{
namespace
{

template <class T>
constexpr std::string_view TypeName()
{
  if constexpr (std::is_same_v<T, std::int32_t>) {
    return "int32_t";
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return "int64_t";
  } else if constexpr (std::is_same_v<T, float>) {
    return "float";
  } else if constexpr (std::is_same_v<T, double>) {
    return "double";
  } else {
    return "unknown";
  }
}

}  // namespace

template <class Index, class Number, std::size_t TileSize>
void DescribeKernelVariant()
{
  std::cout << "KernelVariant<" << TypeName<Index>() << ", " << TypeName<Number>() << ", "
            << TileSize << ">\n";
}

#define SPCRAFT_EXAMPLE_PARAMETER_SEQUENCES \
  ((std::int32_t)(std::int64_t))((float)(double))((32)(64))

#define SPCRAFT_EXAMPLE_INSTANTIATE(r, product) \
  template void DescribeKernelVariant<BOOST_PP_SEQ_ENUM(product)>();

BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_EXAMPLE_INSTANTIATE, SPCRAFT_EXAMPLE_PARAMETER_SEQUENCES)

#define SPCRAFT_EXAMPLE_INVOKE(r, product) DescribeKernelVariant<BOOST_PP_SEQ_ENUM(product)>();

void DescribeAllKernelVariants()
{
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_EXAMPLE_INVOKE, SPCRAFT_EXAMPLE_PARAMETER_SEQUENCES)
}

#undef SPCRAFT_EXAMPLE_INVOKE
#undef SPCRAFT_EXAMPLE_INSTANTIATE
#undef SPCRAFT_EXAMPLE_PARAMETER_SEQUENCES

}  // namespace spcraft::examples

int main()
{
  spcraft::examples::DescribeAllKernelVariants();
  return 0;
}
