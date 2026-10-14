#include <gtest/gtest.h>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "common/BenchmarkInput.h"
#include "common/BenchmarkOptions.h"

namespace spcraft::benchmarks
{
namespace
{

cxxopts::ParseResult Parse(std::vector<std::string> arguments)
{
  arguments.insert(arguments.begin(), "benchmark_options_test");
  std::vector<char*> argv;
  for (auto& argument : arguments) argv.push_back(argument.data());
  cxxopts::Options parser("benchmark_options_test");
  AddCommonOptions(parser);
  const auto result = parser.parse(static_cast<int>(argv.size()), argv.data());
  return result;
}

template <class Body>
void Reject(Body body, const char* message)
{
  try {
    body();
    EXPECT_TRUE((false)) << message;
  } catch (const std::exception&) {
  }
}
}  // namespace
}  // namespace spcraft::benchmarks

namespace spcraft::tests
{
TEST(BenchmarkOptions, Correctness)
{
  using namespace spcraft::benchmarks;
  const auto defaults = ReadBenchmarkOptions(Parse({}));
  EXPECT_TRUE((defaults.max_iterations == 1000 && defaults.max_time_seconds == 10.0))
      << "existing measurement defaults";
  const auto result = Parse({"-i", "7", "--max-time", "0.25", "-o", "raw.json", "--text-report",
                             "report.txt", "-m", "example.mtx", "--precision", "both"});
  const auto benchmark = ReadBenchmarkOptions(result);
  const auto input = ReadMatrixSourceOptions(result);
  const auto types = ReadTypeSelection(result);
  EXPECT_TRUE((benchmark.max_iterations == 7 && benchmark.max_time_seconds == 0.25 &&
               benchmark.output_path == "raw.json" && benchmark.text_report_path == "report.txt" &&
               input.matrix_path == "example.mtx" && types.precision == "both"))
      << "shared and input flags compose in one parser, including short aliases";
  for (const auto& value : {"0", "-1"}) {
    Reject([&] { ReadBenchmarkOptions(Parse({"--iterations", value})); },
           "nonpositive iterations rejected");
  }
  for (const auto& value : {"0", "-1", "nan", "inf"}) {
    Reject([&] { ReadBenchmarkOptions(Parse({"--max-time", value})); },
           "invalid time budget rejected");
  }
  EXPECT_TRUE((ParseThreadCounts(" 1, 4,8 ", 6) == std::vector<int>({1, 4, 8})))
      << "explicit thread list retains order and trims whitespace";
  EXPECT_TRUE((ParseThreadCounts("", 6) == std::vector<int>({1, 2, 4, 6})))
      << "default sweep includes a non-power-of-two maximum";
  EXPECT_TRUE((ParseThreadCounts("", 1) == std::vector<int>({1}))) << "serial default sweep";
  const auto largest = ParseThreadCounts("", std::numeric_limits<int>::max());
  EXPECT_TRUE((largest.front() == 1 && largest.back() == std::numeric_limits<int>::max()))
      << "large default sweep avoids integer overflow";
  for (const auto& value : {"0", "-1", "2oops", "1 2", "99999999999999999999"}) {
    Reject([&] { ParseThreadCounts(value, 8); }, "invalid thread token rejected");
  }
  Reject([] { ReadMatrixSourceOptions(Parse({"--generator", "invalid"})); },
         "unknown generator rejected");
  Reject([] { ReadMatrixSourceOptions(Parse({"--generator", "rmat", "--edge-factor", "-1"})); },
         "negative edge factor rejected before unsigned conversion");
  Reject([] { ReadTypeSelection(Parse({"--precision", "half"})); }, "unknown precision rejected");
}

}  // namespace spcraft::tests
