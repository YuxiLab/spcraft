#include "SpCraft.h"

#include <mpi.h>

#include <iostream>
#include <cxxopts.hpp>

using namespace spcraft;

int ExampleRun()
{
  std::cerr << "Add API usage and reference verification.\n";
  return 1;
}

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0], "Feature usage example");
  options.add_options()("h,help", "Show help");
  const cxxopts::ParseResult arguments = options.parse(argc, argv);
  if (arguments.count("help")) {
    std::cout << options.help() << '\n';
    return 0;
  }
  MPI_Init(&argc, &argv);
  const int status = ExampleRun();
  MPI_Finalize();
  return status;
}
