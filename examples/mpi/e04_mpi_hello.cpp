#ifdef SPCRAFT_USE_MPI

#include <fmt/format.h>
#include <mpi.h>

#include <cstdio>
#include <string>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0;
  int process_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int processor_name_length = 0;
  MPI_Get_processor_name(processor_name, &processor_name_length);
  const std::string processor(processor_name, processor_name_length);

  // Print in rank order so the output is deterministic and easy to read.
  for (int current_rank = 0; current_rank < process_count; ++current_rank) {
    if (rank == current_rank) {
      fmt::print("Hello from rank {} of {} on {}\n", rank, process_count, processor);
      std::fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0) {
    fmt::print("All {} MPI ranks completed.\n", process_count);
  }

  MPI_Finalize();
  return 0;
}

#endif
