#include "utils/MatrixInput.h"

#include <exception>
#include <utility>

#include <cxxopts.hpp>
#include <fmt/format.h>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "core/CooMatrix.h"

namespace spcraft
{
MatrixInput ParseMatrixInput(int argc, const char* const* argv, const char* description)
{
  try {
    cxxopts::Options parser(argv[0], description);
    AddMatrixInputOptions(parser);
    parser.add_options()("h,help", "Print usage");
    parser.positional_help("<matrix.mtx>");
    parser.show_positional_help();
    parser.parse_positional({"matrix"});

    const cxxopts::ParseResult arguments = parser.parse(argc, argv);
    if (arguments.count("help") != 0 || arguments.count("matrix") == 0) {
      fmt::print("{}\n", parser.help());
      return {std::nullopt, arguments.count("help") != 0 ? 0 : 1};
    }
    return {ReadMatrixInputOptions(arguments), 0};
  } catch (const std::exception& error) {
    fmt::print(stderr, "{}\n", error.what());
    return {std::nullopt, 1};
  }
}

template <class IT, class NT, class OT>
CooMatrix<IT, NT, OT> ReadMatrixInput(const std::string& path)
{
  CooMatrix<IT, NT, OT> matrix(path);
  return matrix;
}

#define SPCRAFT_DEFINE_MATRIX_INPUT(r, product)                                                          \
  template CooMatrix<BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), BOOST_PP_SEQ_ELEM(1, product), \
                     BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product))>                                \
  ReadMatrixInput<BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), BOOST_PP_SEQ_ELEM(1, product),    \
                  BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product))>(const std::string&);

// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_MATRIX_INPUT,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)))
// clang-format on
#undef SPCRAFT_DEFINE_MATRIX_INPUT

}  // namespace spcraft
