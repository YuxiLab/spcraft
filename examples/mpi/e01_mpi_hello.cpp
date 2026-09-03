#include <mpi.h>

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0;
  int process_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_length = 0;
  MPI_Get_processor_name(processor_name, &name_length);

  std::cout << "Hello from rank " << rank << " of " << process_count << " on "
            << std::string(processor_name, name_length) << '\n';

  MPI_Finalize();
  return 0;
}
