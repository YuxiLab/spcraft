#include "SpCraft.h"

#include <mpi.h>

#include <iostream>
#include <benchmark/benchmark.h>

using namespace spcraft;

int BenchmarkRun(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  // Register the feature benchmark after preparing and verifying its inputs.
  std::cerr << "Add a verified benchmark before collecting timings.\n";
  benchmark::Shutdown();
  return 1;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  const int status = ::BenchmarkRun(argc, argv);
  MPI_Finalize();
  return status;
}
