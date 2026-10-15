#include "mkl/MKLOptions.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cxxopts.hpp>
#include <mkl.h>

#include "utils/omp/omp_wrapper.h"

namespace spcraft::benchmarks
{
std::optional<MKLSpMVOptions> ParseMKLSpMVOptions(int argc, char** argv)
{
  static_assert(sizeof(MKL_INT) == sizeof(std::int32_t) || sizeof(MKL_INT) == sizeof(std::int64_t));
  const char* integer_type = sizeof(MKL_INT) == sizeof(std::int32_t) ? "int32" : "int64";
  cxxopts::Options parser("spmv_mkl_benchmark", "oneMKL SpMV benchmark over a thread-count sweep");
  AddBenchmarkOptions(parser);
  AddBenchmarkInputOptions(parser, integer_type, integer_type);
  const auto result = parser.parse(argc, argv);
  if (result.count("help") > 0) {
    std::cout << parser.help() << '\n';
    return std::nullopt;
  }

  MKLSpMVOptions options;
  options.benchmark = ReadBenchmarkOptions(result);
  options.input.matrix = ReadMatrixSourceOptions(result);
  options.input.types = ReadTypeSelection(result);
  options.input.thread_counts = ParseThreadCounts(result["threads"].as<std::string>(), OMP_GET_MAX_THREADS());
  if ((options.input.types.index != "both" && options.input.types.index != integer_type) ||
      (options.input.types.offset != "both" && options.input.types.offset != integer_type)) {
    throw std::invalid_argument(std::string("this oneMKL build requires --index and --offset to select ") +
                                integer_type + " (MKL_INT)");
  }
  const auto& matrix = options.input.matrix;
  if (!matrix.matrix_path) {
    if (matrix.generator == "er" && matrix.vertices > std::numeric_limits<MKL_INT>::max()) {
      throw std::invalid_argument("vertices exceed this oneMKL build's MKL_INT range");
    }
    if (matrix.generator == "rmat" && matrix.scale >= std::numeric_limits<MKL_INT>::digits) {
      throw std::invalid_argument("R-MAT vertex count exceeds this oneMKL build's MKL_INT range");
    }
  }
  return options;
}
}  // namespace spcraft::benchmarks
