#include "SpCraft.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <utility>

#include <benchmark/benchmark.h>

using namespace spcraft;

template <class Matrix>
void RegisterCloneBenchmark(const char* name, const Matrix& matrix)
{
  constexpr int kIterations = 100;
  const auto measure = [&matrix](benchmark::State& state) {
    for (auto iteration : state) {
      MPI_Barrier(matrix.commgrid->World());
      const double start = MPI_Wtime();
      {
        Matrix copied = matrix.Clone();
        benchmark::DoNotOptimize(copied.local.nnz);
        benchmark::ClobberMemory();
      }
      const double elapsed = MPI_Wtime() - start;
      double slowest = 0;
      MPI_Allreduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, matrix.commgrid->World());
      state.SetIterationTime(slowest);
    }
  };
  benchmark::RegisterBenchmark(name, measure)->Iterations(kIterations)->UseManualTime();
}

int BenchmarkRun(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  OMP_SET_NUM_THREADS(1);

  // Build a sparse diagonal local block before measuring copies of each format.
  using IT = int64_t;
  using NT = double;
  using OT = int64_t;
  constexpr IT kLocalExtent = 4096;
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const IT m = kLocalExtent * grid->Rows();
  const IT n = kLocalExtent * grid->Cols();
  DistSpCOO<IT, NT, OT> coo(grid, m, n);
  coo.Allocate(kLocalExtent);
  for (IT index = 0; index < kLocalExtent; ++index) coo.local.entries[index] = {index, index, NT{1}};
  CsrMatrix<IT, NT, OT> local_csr = coo.local.ToCsr();
  CscMatrix<IT, NT, OT> local_csc = coo.local.ToCsc();
  const DistSpCSR<IT, NT, OT> csr(grid, std::move(local_csr), m, n);
  const DistSpCSC<IT, NT, OT> csc(grid, std::move(local_csc), m, n);
  DcscMatrix<IT, NT, OT> local_dcsc = DcscMatrix<IT, NT, OT>::FromCsc(csc.local);
  const DistSpDCSC<IT, NT, OT> dcsc(grid, std::move(local_dcsc), m, n);

  // Check copy ownership and sample values before registration; setup is not timed.
  const DistSpCOO<IT, NT, OT> check_coo = coo.Clone();
  const DistSpCSR<IT, NT, OT> check_csr = csr.Clone();
  const DistSpCSC<IT, NT, OT> check_csc = csc.Clone();
  const DistSpDCSC<IT, NT, OT> check_dcsc = dcsc.Clone();
  const IT last = kLocalExtent - 1;
  const bool matches = check_coo.local.entries != coo.local.entries && check_csr.local.val != csr.local.val &&
                       check_csc.local.val != csc.local.val && check_coo.local.entries[last].val == NT{1} &&
                       check_csr.local.val[last] == NT{1} && check_csc.local.val[last] == NT{1} &&
                       check_dcsc.local.val != dcsc.local.val && check_dcsc.local.val[last] == NT{1};
  const int local_status = matches ? 0 : 1;
  int status = 0;
  MPI_Allreduce(&local_status, &status, 1, MPI_INT, MPI_MAX, grid->World());
  if (status != 0) return status;
  RegisterCloneBenchmark("DistSpCOO/CloneAndRelease", coo);
  RegisterCloneBenchmark("DistSpCSR/CloneAndRelease", csr);
  RegisterCloneBenchmark("DistSpCSC/CloneAndRelease", csc);
  RegisterCloneBenchmark("DistSpDCSC/CloneAndRelease", dcsc);
  std::ostringstream hidden;
  benchmark::ConsoleReporter reporter;
  if (grid->Rank() != 0) {
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
