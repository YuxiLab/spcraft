#pragma once

#include "SpCraft.h"

#include <sstream>

#include <benchmark/benchmark.h>

using namespace spcraft;

using SummaBenchmarkCSC = DistSpCSC<int64_t, double, int64_t>;
using SummaBenchmarkCOO = DistSpCOO<int64_t, double, int64_t>;
constexpr int64_t kSummaExtent = 2048;
constexpr int kSummaIterations = 20;

//! Construct a distributed diagonal through public COO and CSC APIs before timing.
template <class Predicate>
SummaBenchmarkCSC SummaBenchmarkInput(std::shared_ptr<const CommGrid2D> grid, Predicate owns)
{
  SummaBenchmarkCOO selected(grid, kSummaExtent, kSummaExtent);
  int64_t count = 0;
  for (int64_t i = 0; i < kSummaExtent; ++i)
    if (owns(i)) ++count;
  selected.Allocate(count);
  int64_t position = 0;
  for (int64_t i = 0; i < kSummaExtent; ++i) {
    if (owns(i)) selected.local.entries[position++] = {selected.LocalRow(i), selected.LocalCol(i), 2};
  }
  CscMatrix<int64_t, double, int64_t> local = selected.local.ToCsc();
  SummaBenchmarkCSC result(grid, std::move(local), kSummaExtent, kSummaExtent);
  return result;
}

//! Independent closed form: (2I)*(2I)=4I, including local ownership and nnz.
inline bool VerifySummaBenchmark(const SummaBenchmarkCOO& product)
{
  int64_t expected = 0;
  for (int64_t i = 0; i < kSummaExtent; ++i)
    if (product.Owns(i, i)) ++expected;
  if (product.local.nnz != expected) return false;
  for (int64_t p = 0; p < product.local.nnz; ++p) {
    const TupleEntry<int64_t, double>& entry = product.local.entries[p];
    if (product.GlobalRow(entry.row) != product.GlobalCol(entry.col) || entry.val != 4) return false;
  }
  return true;
}

inline void ReportSummaBenchmarks(MPI_Comm world)
{
  int rank = 0;
  MPI_Comm_rank(world, &rank);
  std::ostringstream hidden;
  benchmark::ConsoleReporter reporter;
  if (rank != 0) {
    reporter.SetOutputStream(&hidden);
    reporter.SetErrorStream(&hidden);
  }
  benchmark::RunSpecifiedBenchmarks(&reporter);
  benchmark::ClearRegisteredBenchmarks();
  benchmark::Shutdown();
}
