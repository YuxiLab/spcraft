#include "common/BenchmarkInput.h"
#include "common/BenchmarkOptions.h"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include <cxxopts.hpp>

namespace spcraft::benchmarks
{
void AddBenchmarkInputOptions(cxxopts::Options& parser, const char* default_index, const char* default_offset)
{
  const MatrixSourceOptions defaults;
  // clang-format off
  parser.add_options("Input")
      ("m,matrix", "Path to a Matrix Market (.mtx) file; overrides --generator", cxxopts::value<std::string>())
      ("generator", "Generated graph: 'er' or 'rmat'", cxxopts::value<std::string>()->default_value(defaults.generator))
      ("v,vertices", "Vertex count for the generated ER graph", cxxopts::value<std::int64_t>()->default_value(std::to_string(defaults.vertices)))
      ("d,degree", "Expected degree for the generated ER graph", cxxopts::value<double>()->default_value(std::to_string(defaults.degree)))
      ("scale", "R-MAT log2 vertex count", cxxopts::value<std::int64_t>()->default_value(std::to_string(defaults.scale)))
      ("edge-factor", "R-MAT sampled edges per vertex", cxxopts::value<std::int64_t>()->default_value(std::to_string(defaults.edge_factor)))
      ("p,precision", "Numeric type: 'float', 'double', or 'both'", cxxopts::value<std::string>()->default_value("double"))
      ("index", "Index type: 'int32', 'int64', or 'both'", cxxopts::value<std::string>()->default_value(default_index))
      ("offset", "Row-offset type: 'int32', 'int64', or 'both'", cxxopts::value<std::string>()->default_value(default_offset))
      ("t,threads", "Comma-separated thread counts, e.g. '1,2,4,8'; empty means powers of two up to the maximum", cxxopts::value<std::string>()->default_value(""))
      ("s,seed", "Random seed for the graph generator and input vector", cxxopts::value<int>()->default_value(std::to_string(defaults.seed)))
      ("h,help", "Print help");
  // clang-format on
}

void AddCommonOptions(cxxopts::Options& parser, const char* default_index, const char* default_offset)
{
  AddBenchmarkOptions(parser);
  AddBenchmarkInputOptions(parser, default_index, default_offset);
}

MatrixSourceOptions ReadMatrixSourceOptions(const cxxopts::ParseResult& result)
{
  MatrixSourceOptions options;
  if (result.count("matrix") > 0) {
    options.matrix_path = result["matrix"].as<std::string>();
    if (options.matrix_path->empty()) {
      throw std::invalid_argument("matrix path must not be empty");
    }
  }
  options.generator = result["generator"].as<std::string>();
  options.vertices = result["vertices"].as<std::int64_t>();
  options.degree = result["degree"].as<double>();
  options.scale = result["scale"].as<std::int64_t>();
  options.edge_factor = result["edge-factor"].as<std::int64_t>();
  options.seed = result["seed"].as<int>();
  // File input overrides generator settings, including values unused by that source.
  if (!options.matrix_path) {
    if (options.generator == "er") {
      if (options.vertices <= 0 || !std::isfinite(options.degree) || options.degree < 0.0) {
        throw std::invalid_argument("ER vertices must be positive and degree finite and nonnegative");
      }
    } else if (options.generator == "rmat") {
      if (options.scale <= 0 || options.edge_factor <= 0) {
        throw std::invalid_argument("R-MAT scale and edge-factor must be positive");
      }
    } else {
      throw std::invalid_argument("generator must be 'er' or 'rmat'");
    }
  }
  return options;
}

TypeSelection ReadTypeSelection(const cxxopts::ParseResult& result)
{
  TypeSelection options;
  options.precision = result["precision"].as<std::string>();
  options.index = result["index"].as<std::string>();
  options.offset = result["offset"].as<std::string>();
  if (options.precision != "float" && options.precision != "double" && options.precision != "both") {
    throw std::invalid_argument("precision must be 'float', 'double', or 'both'");
  }
  for (const auto* choice : {&options.index, &options.offset}) {
    if (*choice != "int32" && *choice != "int64" && *choice != "both") {
      throw std::invalid_argument("index and offset types must be 'int32', 'int64', or 'both'");
    }
  }
  return options;
}

std::vector<int> ParseThreadCounts(const std::string& argument, int maximum_threads)
{
  std::vector<int> threads;
  std::string_view remaining(argument);
  while (!remaining.empty()) {
    const auto comma = remaining.find(',');
    auto token = remaining.substr(0, comma);
    const auto first = token.find_first_not_of(" \t");
    if (first != std::string_view::npos) {
      token = token.substr(first, token.find_last_not_of(" \t") - first + 1);
      int count = 0;
      const auto parsed = std::from_chars(token.data(), token.data() + token.size(), count);
      if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || count <= 0) {
        throw std::invalid_argument("thread counts must be positive integers");
      }
      threads.push_back(count);
    }
    if (comma == std::string_view::npos) break;
    remaining.remove_prefix(comma + 1);
  }
  if (threads.empty()) {
    for (int count = 1; count < maximum_threads;) {
      threads.push_back(count);
      // Avoid overflowing int when the maximum is not a power of two.
      if (count > maximum_threads / 2) break;
      count *= 2;
    }
    threads.push_back(maximum_threads);
  }
  return threads;
}
}  // namespace spcraft::benchmarks
