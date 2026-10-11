#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cxxopts
{
class Options;
class ParseResult;
}  // namespace cxxopts

namespace spcraft::benchmarks
{
struct MatrixSourceOptions {
  std::optional<std::string> matrix_path;
  std::string generator = "er";
  std::int64_t vertices = 50000;
  double degree = 16.0;
  std::int64_t scale = 16;
  std::int64_t edge_factor = 16;
  int seed = 42;
};

struct TypeSelection {
  std::string precision = "double";
  std::string index = "int32";
  std::string offset = "int64";
};

// Backend modules choose their defaults while sharing the input flag definitions.
void AddBenchmarkInputOptions(cxxopts::Options& parser, const char* default_index,
                              const char* default_offset);
MatrixSourceOptions ReadMatrixSourceOptions(const cxxopts::ParseResult& result);
TypeSelection ReadTypeSelection(const cxxopts::ParseResult& result);
std::vector<int> ParseThreadCounts(const std::string& argument, int maximum_threads);

// Existing drivers can compose the same groups before adopting typed options.
void AddCommonOptions(cxxopts::Options& parser, const char* default_index = "int32",
                      const char* default_offset = "int64");
}  // namespace spcraft::benchmarks
