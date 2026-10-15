#include "SpCraft.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

#include <cxxopts.hpp>

using namespace spcraft;

int ExampleRun(int rows, int cols, bool column_major)
{
  // Empty callbacks select row-major rank order. A callback may instead assign
  // any bijection between source ranks and process coordinates.
  CommGrid2D::RankLayout layout;
  if (column_major) {
    layout = [](int rank, int grid_rows, int) {
      return std::pair<int, int>{rank % grid_rows, rank / grid_rows};
    };
  }
  const CommGrid2D grid(MPI_COMM_WORLD, rows, cols, layout);

  // Rank in a row communicator equals the process column, regardless of layout.
  int received = grid.Rank();
  MPI_Bcast(&received, 1, MPI_INT, 0, grid.RowWorld());
  const int expected = grid.RankAt(grid.Row(), 0);
  const int local_status = received == expected ? 0 : 1;
  int status = 0;
  MPI_Allreduce(&local_status, &status, 1, MPI_INT, MPI_MAX, grid.World());
  if (grid.Rank() == 0) {
    std::cout << "Process grid: " << grid.Rows() << " x " << grid.Cols() << '\n';
    for (int row = 0; row < grid.Rows(); ++row) {
      for (int col = 0; col < grid.Cols(); ++col) {
        std::cout << "(" << row << ", " << col << ") -> rank " << grid.RankAt(row, col) << '\n';
      }
    }
    std::cout << "Row broadcasts " << (status == 0 ? "passed" : "failed") << '\n';
  }
  // The grid releases its communicators when this function returns, before Finalize.
  return status;
}

int main(int argc, char** argv)
{
  cxxopts::Options parser(argv[0], "2D process ownership and row/column communicators");
  cxxopts::OptionAdder flags = parser.add_options();
  const std::shared_ptr<cxxopts::Value> rows_option = cxxopts::value<int>()->default_value("0");
  const std::shared_ptr<cxxopts::Value> cols_option = cxxopts::value<int>()->default_value("0");
  flags("rows", "Process rows (0: choose automatically)", rows_option);
  flags("cols", "Process columns (0: choose automatically)", cols_option);
  flags("column-major", "Assign source ranks in column-major order");
  flags("h,help", "Show help");
  MPI_Init(&argc, &argv);
  int status = 0;
  try {
    const cxxopts::ParseResult arguments = parser.parse(argc, argv);
    if (arguments.count("help")) {
      std::cout << parser.help() << '\n';
    } else {
      const int rows = arguments["rows"].as<int>();
      const int cols = arguments["cols"].as<int>();
      const bool column_major = arguments.count("column-major") != 0;
      status = ExampleRun(rows, cols, column_major);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  MPI_Finalize();
  return status;
}
