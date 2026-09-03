#include <mpi.h>

#include <iostream>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0;
  int process_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  if (process_count < 2) {
    if (rank == 0) {
      std::cerr << "Run this example with at least two MPI processes.\n";
    }
    MPI_Finalize();
    return 1;
  }

  constexpr int message_tag = 42;
  if (rank == 0) {
    const int value = 1234;
    MPI_Send(&value, 1, MPI_INT, 1, message_tag, MPI_COMM_WORLD);
    std::cout << "Rank 0 sent " << value << " to rank 1.\n";
  } else if (rank == 1) {
    int value = 0;
    MPI_Status status;
    MPI_Recv(&value, 1, MPI_INT, 0, message_tag, MPI_COMM_WORLD, &status);
    std::cout << "Rank 1 received " << value << " from rank " << status.MPI_SOURCE << ".\n";
  }

  MPI_Finalize();
  return 0;
}
