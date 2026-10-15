#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/BenchmarkOptions.h"
#include "mkl/MKLOptions.h"

namespace spcraft::benchmarks
{
namespace
{

std::optional<MKLSpMVOptions> Parse(std::vector<std::string> arguments)
{
  arguments.insert(arguments.begin(), "bench_mkl_spmv");
  std::vector<char*> argv;
  for (auto& argument : arguments) argv.push_back(argument.data());
  const auto options = ParseMKLSpMVOptions(static_cast<int>(argv.size()), argv.data());
  return options;
}

void Reject(std::vector<std::string> arguments, const char* message)
{
  try {
    Parse(std::move(arguments));
    EXPECT_TRUE((false)) << message;
  } catch (const std::exception&) {
  }
}
}  // namespace
}  // namespace spcraft::benchmarks

namespace spcraft::tests
{
TEST(MklOptions, Correctness)
{
  using namespace spcraft::benchmarks;
  const auto defaults = Parse({});
  EXPECT_TRUE((defaults && defaults->input.types.index == defaults->input.types.offset &&
               !defaults->input.thread_counts.empty() && defaults->input.matrix.seed == 42))
      << "MKL defaults select its interface width and initialize the thread sweep";
  const auto file = Parse({"--matrix",   "A.mtx",    "--generator",   "unused",     "--iterations", "12",
                           "--max-time", "0.5",      "--threads",     "1,3",        "--precision",  "both",
                           "--index",    "both",     "--offset",      "both",       "--seed",       "9",
                           "--output",   "raw.json", "--text-report", "summary.txt"});
  EXPECT_TRUE(
      (file && file->benchmark.max_iterations == 12 && file->benchmark.max_time_seconds == 0.5 &&
       file->benchmark.output_path == "raw.json" && file->benchmark.text_report_path == "summary.txt" &&
       file->input.matrix.matrix_path == "A.mtx" && file->input.matrix.seed == 9 &&
       file->input.types.precision == "both" && file->input.thread_counts == std::vector<int>({1, 3})))
      << "one parse composes both groups and file input overrides generator selection";
  const auto graph = Parse({"--generator", "rmat", "--scale", "8", "--edge-factor", "4"});
  EXPECT_TRUE((graph && !graph->input.matrix.matrix_path && graph->input.matrix.scale == 8 &&
               graph->input.matrix.edge_factor == 4))
      << "R-MAT input settings retained";
  const std::string unsupported = defaults->input.types.index == "int32" ? "int64" : "int32";
  Reject({"--index", unsupported}, "unsupported MKL column width rejected");
  Reject({"--offset", unsupported}, "unsupported MKL row-offset width rejected");
  Reject({"--iterations", "0"}, "common validation runs in the MKL parser");
  Reject({"--threads", "2x"}, "invalid thread list rejected");
  Reject({"--matrix", ""}, "empty matrix filename rejected");
  Reject({"--generator", "rmat", "--scale", "63"}, "unrepresentable generated shape rejected");
  Reject({"--not-an-option"}, "unknown option rejected");

  std::ostringstream output;
  auto* previous = std::cout.rdbuf(output.rdbuf());
  const auto help = Parse({"--help", "--iterations", "0", "--index", unsupported});
  std::cout.rdbuf(previous);
  EXPECT_TRUE((!help && output.str().find("--iterations") != std::string::npos &&
               output.str().find("--matrix") != std::string::npos &&
               output.str().find("--threads") != std::string::npos))
      << "combined help includes both groups and exits before runtime validation";
}

}  // namespace spcraft::tests
