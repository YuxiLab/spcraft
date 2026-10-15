#include "SpCraft.h"

#include <sstream>

#include <benchmark/benchmark.h>

using namespace spcraft;

void MeasureGridCreation(benchmark::State& state)
{
  for (auto iteration : state) {
    // Fixed iteration counts keep all MPI ranks in the same collective sequence.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    {
      const CommGrid2D grid;
      benchmark::DoNotOptimize(grid.World());
    }
    const double elapsed = MPI_Wtime() - start;
    double slowest = 0;
    MPI_Allreduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    state.SetIterationTime(slowest);
  }
}

int BenchmarkRun(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  // Verify communicator membership once, outside the measured loop.
  {
    const CommGrid2D grid;
    int row_size = 0;
    MPI_Comm_size(grid.RowWorld(), &row_size);
    if (row_size != grid.Cols()) return 1;
  }
  constexpr int kIterations = 100;
  benchmark::RegisterBenchmark("CommGrid2D/CreateAndRelease", MeasureGridCreation)
      ->Iterations(kIterations)
      ->UseManualTime();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::ostringstream hidden;
  benchmark::ConsoleReporter reporter;
  if (rank != 0) {
    reporter.SetOutputStream(&hidden);
    reporter.SetErrorStream(&hidden);
  }
  benchmark::RunSpecifiedBenchmarks(&reporter);
  benchmark::Shutdown();
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  const int status = ::BenchmarkRun(argc, argv);
  MPI_Finalize();
  return status;
}
