#include "SpCraft.h"
#include "SummaBenchmarkCommon.h"

#include <iostream>
#include <string>
#include <vector>

#include <cxxopts.hpp>

using namespace spcraft;

int BenchmarkRun(int argc, char** argv, int layers)
{
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  OMP_SET_NUM_THREADS(1);
  const CommGrid3D grid(MPI_COMM_WORLD, layers);
  int rank = 0;
  MPI_Comm_rank(grid.World(), &rank);
  const int side = grid.Plane()->Cols();
  const int64_t block = kSummaExtent / side + (kSummaExtent % side != 0);
  const auto owns = [&](int64_t i) { return grid.AOwner(i / block, i / block) == rank; };
  const SummaBenchmarkCSC a = SummaBenchmarkInput(grid.Plane(), owns);
  const SummaBenchmarkCOO checked = Mpi3DSummaSpGEMM<PlusTimesRing<double>>(a, a, grid);
  int failed = grid.Layer() == 0 && !VerifySummaBenchmark(checked) ? 1 : 0;
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, grid.World());
  if (any_failed) return 1;

  // Input loading, partitioning, and verification stay outside each timed iteration.
  const auto measure = [&](benchmark::State& state) {
    for (auto iteration : state) {
      MPI_Barrier(grid.World());
      const double start = MPI_Wtime();
      {
        SummaBenchmarkCOO product = Mpi3DSummaSpGEMM<PlusTimesRing<double>>(a, a, grid);
        benchmark::DoNotOptimize(product.local.nnz);
      }
      const double elapsed = MPI_Wtime() - start;
      double slowest = 0;
      MPI_Allreduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, grid.World());
      state.SetIterationTime(slowest);
    }
  };
  benchmark::RegisterBenchmark("SpGEMM/Summa3D/CSC", measure)->Iterations(kSummaIterations)->UseManualTime();
  ReportSummaBenchmarks(grid.World());
  return 0;
}

int main(int argc, char** argv)
{
  cxxopts::Options parser(argv[0], "3D SUMMA benchmark; also accepts Google Benchmark flags");
  parser.allow_unrecognised_options();
  parser.add_options()("layers", "Layer count", cxxopts::value<int>())("h,help", "Show help");
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int status = 0;
  try {
    if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI needs FUNNELED threading");
    const cxxopts::ParseResult args = parser.parse(argc, argv);
    if (args.count("help"))
      std::cout << parser.help() << '\n';
    else {
      const int layers = args["layers"].as<int>();
      std::vector<std::string> remaining{argv[0]};
      const std::vector<std::string> unmatched = args.unmatched();
      remaining.insert(remaining.end(), unmatched.begin(), unmatched.end());
      std::vector<char*> benchmark_argv;
      for (std::string& argument : remaining) benchmark_argv.push_back(argument.data());
      benchmark_argv.push_back(nullptr);
      const int benchmark_argc = static_cast<int>(remaining.size());
      status = ::BenchmarkRun(benchmark_argc, benchmark_argv.data(), layers);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  MPI_Finalize();
  return status;
}
