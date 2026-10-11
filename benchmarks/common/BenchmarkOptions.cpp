#include "common/BenchmarkOptions.h"

#include <cmath>
#include <stdexcept>

#include <cxxopts.hpp>

namespace spcraft::benchmarks
{
void AddBenchmarkOptions(cxxopts::Options& parser)
{
  const BenchmarkOptions defaults;
  // clang-format off
  parser.add_options("Benchmark")
      ("i,iterations", "Maximum timed iterations per configuration",
       cxxopts::value<int>()->default_value(std::to_string(defaults.max_iterations)))
      ("max-time", "Wall-clock budget per configuration in seconds",
       cxxopts::value<double>()->default_value(std::to_string(defaults.max_time_seconds)))
      ("o,output", "Write the raw per-iteration log to this JSON file",
       cxxopts::value<std::string>()->default_value(""))
      ("text-report", "Write the human-readable report to this text file",
       cxxopts::value<std::string>()->default_value(""));
  // clang-format on
}

BenchmarkOptions ReadBenchmarkOptions(const cxxopts::ParseResult& result)
{
  BenchmarkOptions options;
  options.max_iterations = result["iterations"].as<int>();
  options.max_time_seconds = result["max-time"].as<double>();
  options.output_path = result["output"].as<std::string>();
  options.text_report_path = result["text-report"].as<std::string>();
  if (options.max_iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (!std::isfinite(options.max_time_seconds) || options.max_time_seconds <= 0.0) {
    throw std::invalid_argument("max-time must be finite and positive");
  }
  return options;
}
}  // namespace spcraft::benchmarks
