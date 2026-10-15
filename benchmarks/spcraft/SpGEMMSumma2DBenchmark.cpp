#include "SpCraft.h"
#include "SummaBenchmarkCommon.h"

using namespace spcraft;

int BenchmarkRun(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  OMP_SET_NUM_THREADS(1);
  const std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const int64_t row_extent = kSummaExtent / grid->Rows() + (kSummaExtent % grid->Rows() != 0);
  const int64_t col_extent = kSummaExtent / grid->Cols() + (kSummaExtent % grid->Cols() != 0);
  const auto owns = [&](int64_t i) { return grid->Owner(i / row_extent, i / col_extent) == grid->Rank(); };
  const SummaBenchmarkCSC a = SummaBenchmarkInput(grid, owns);
  const SummaBenchmarkCOO checked = Mpi2DSummaSpGEMM<PlusTimesRing<double>>(a, a);
  int failed = VerifySummaBenchmark(checked) ? 0 : 1;
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, grid->World());
  if (any_failed) return 1;

  // Fixed counts keep all ranks in the same collective sequence; time the slowest rank.
  const auto measure = [&](benchmark::State& state) {
    for (auto iteration : state) {
      MPI_Barrier(grid->World());
      const double start = MPI_Wtime();
      {
        SummaBenchmarkCOO product = Mpi2DSummaSpGEMM<PlusTimesRing<double>>(a, a);
        benchmark::DoNotOptimize(product.local.entries);
      }
      const double elapsed = MPI_Wtime() - start;
      double slowest = 0;
      MPI_Allreduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, grid->World());
      state.SetIterationTime(slowest);
    }
  };
  benchmark::RegisterBenchmark("SpGEMM/Summa2D/CSC", measure)->Iterations(kSummaIterations)->UseManualTime();
  ReportSummaBenchmarks(grid->World());
  return 0;
}

int main(int argc, char** argv)
{
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  const int status = provided < MPI_THREAD_FUNNELED ? 1 : ::BenchmarkRun(argc, argv);
  MPI_Finalize();
  return status;
}
