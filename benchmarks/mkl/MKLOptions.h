#pragma once

#include <optional>
#include <vector>

#include "common/BenchmarkInput.h"
#include "common/BenchmarkOptions.h"

namespace spcraft::benchmarks
{
struct MKLOptions {
  MatrixSourceOptions matrix;
  TypeSelection types;
  std::vector<int> thread_counts;
};

struct MKLSpMVOptions {
  BenchmarkOptions benchmark;
  MKLOptions input;
};

// Prints combined help and returns nullopt for --help; invalid arguments throw.
std::optional<MKLSpMVOptions> ParseMKLSpMVOptions(int argc, char** argv);
}  // namespace spcraft::benchmarks
