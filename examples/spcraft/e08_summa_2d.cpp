#include "SpCraft.h"
#include "SummaExampleCommon.h"

#include <iostream>

#include <cxxopts.hpp>

using namespace spcraft;

template <class IT, class NT, class OT>
int ExampleRun(const MatrixInputOptions& input, int grid_rows, int grid_cols)
{
  // Each rank reads the teaching input; production loaders can distribute directly.
  const CooMatrix<IT, NT, OT> source = ReadMatrixInput<IT, NT, OT>(input.matrix_path);
  if (source.m != source.n) throw std::invalid_argument("This A*A example needs a square matrix");
  const MPI_Comm world = MPI_COMM_WORLD;
  const std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>(world, grid_rows, grid_cols);

  // The chosen process grid determines each rank's rectangular matrix block.
  const IT block_rows = source.m / grid->Rows() + (source.m % grid->Rows() != 0);
  const IT block_cols = source.n / grid->Cols() + (source.n % grid->Cols() != 0);
  const IT row_extent = std::max(IT{1}, block_rows);
  const IT col_extent = std::max(IT{1}, block_cols);
  const auto owns = [&](IT row, IT col) {
    return grid->Owner(row / row_extent, col / col_extent) == grid->Rank();
  };
  const DistSpCSC<IT, NT, OT> a = PartitionSummaInput(source, grid, owns);

  // The collective API takes distributed matrices, including their grid and global dimensions.
  const DistSpCOO<IT, NT, OT> product = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(a, a);
  const ExampleSparseMatrix<IT, NT> reference = ToEigenReference(source);
  const ExampleSparseMatrix<IT, NT> expected = reference * reference;
  const int local_status = CheckSummaReference(product, expected) ? 0 : 1;
  int status = 0;
  MPI_Allreduce(&local_status, &status, 1, MPI_INT, MPI_MAX, grid->World());
  if (grid->Rank() == 0) {
    std::cout << "Synchronous 2D SUMMA Eigen check " << (status == 0 ? "passed" : "failed") << '\n';
  }
  return status;
}

int main(int argc, char** argv)
{
  cxxopts::Options parser(argv[0], "Synchronous 2D SUMMA on a chosen process grid: C=A*A");
  AddMatrixInputOptions(parser);
  cxxopts::OptionAdder flags = parser.add_options();
  const std::shared_ptr<cxxopts::Value> rows_flag = cxxopts::value<int>()->default_value("0");
  const std::shared_ptr<cxxopts::Value> cols_flag = cxxopts::value<int>()->default_value("0");
  flags("grid-rows", "Process rows (0 lets MPI choose)", rows_flag);
  flags("grid-cols", "Process columns (0 lets MPI choose)", cols_flag);
  flags("h,help", "Show help");
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int status = 0;
  try {
    if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI needs FUNNELED threading");
    const cxxopts::ParseResult args = parser.parse(argc, argv);
    if (args.count("help"))
      std::cout << parser.help() << '\n';
    else {
      const MatrixInputOptions input = ReadMatrixInputOptions(args);
      const int grid_rows = args["grid-rows"].as<int>();
      const int grid_cols = args["grid-cols"].as<int>();
      const auto run = [&]<class IT, class NT, class OT>() {
        return ExampleRun<IT, NT, OT>(input, grid_rows, grid_cols);
      };
      status = DispatchMatrixInputTypes(input, run);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  MPI_Finalize();
  return status;
}
