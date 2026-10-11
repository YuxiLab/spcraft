#pragma once

#include <string>

namespace cxxopts
{
class Options;
class ParseResult;
}  // namespace cxxopts

namespace spcraft::benchmarks
{
struct BenchmarkOptions {
  int max_iterations = 1000;
  double max_time_seconds = 10.0;
  std::string output_path;
  std::string text_report_path;
};

// Compose option groups in a parser implementation, then read them after one parse.
void AddBenchmarkOptions(cxxopts::Options& parser);
BenchmarkOptions ReadBenchmarkOptions(const cxxopts::ParseResult& result);
}  // namespace spcraft::benchmarks
