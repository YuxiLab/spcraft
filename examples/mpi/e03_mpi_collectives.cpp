#include <mpi.h>

#include <iostream>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0;
  int process_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  int shared_value = rank == 0 ? 10 : 0;
  MPI_Bcast(&shared_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::cout << "Rank " << rank << " received broadcast value " << shared_value << ".\n";

  const int local_value = rank + 1;
  int global_sum = 0;
  MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const int expected_sum = process_count * (process_count + 1) / 2;
    std::cout << "Reduced sum: " << global_sum << " (expected " << expected_sum << ").\n";
  }

  MPI_Finalize();
  return 0;
}
