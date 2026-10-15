#include "SpCraft.h"
#include "DistributedExampleCommon.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <cxxopts.hpp>

using namespace spcraft;

template <class IT, class NT, class OT>
int ExampleRun(const MatrixInputOptions& input, int grid_rows, int grid_cols)
{
  // For teaching, each rank reads the same input and selects its owned entries.
  // Production input loaders can distribute directly into each rank's local matrix.
  const CooMatrix<IT, NT, OT> source = ReadMatrixInput<IT, NT, OT>(input.matrix_path);
  const MPI_Comm world = MPI_COMM_WORLD;
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>(world, grid_rows, grid_cols);
  DistSpCOO<IT, NT, OT> coo(grid, source.m, source.n);
  OT count = 0;
  for (OT position = 0; position < source.nnz; ++position) {
    const TupleEntry<IT, NT>& entry = source.entries[position];
    if (coo.Owns(entry.row, entry.col)) ++count;
  }
  coo.Allocate(count);
  OT next = 0;
  for (OT position = 0; position < source.nnz; ++position) {
    const TupleEntry<IT, NT>& entry = source.entries[position];
    if (coo.Owns(entry.row, entry.col)) {
      coo.local.entries[next++] = {coo.LocalRow(entry.row), coo.LocalCol(entry.col), entry.val};
    }
  }

  // Convert local storage through the public API; every format retains the grid.
  CsrMatrix<IT, NT, OT> local_csr = coo.local.ToCsr();
  CscMatrix<IT, NT, OT> local_csc = coo.local.ToCsc();
  const DistSpCSR<IT, NT, OT> csr(grid, std::move(local_csr), source.m, source.n);
  const DistSpCSC<IT, NT, OT> csc(grid, std::move(local_csc), source.m, source.n);
  DcscMatrix<IT, NT, OT> local_dcsc = DcscMatrix<IT, NT, OT>::FromCsc(csc.local);
  const DistSpDCSC<IT, NT, OT> dcsc(grid, std::move(local_dcsc), source.m, source.n);
  const DistSpCOO<IT, NT, OT> copied = coo.Clone();

  // Check global ownership against Eigen and check each local representation.
  const ExampleSparseMatrix<IT, NT> reference = ToEigenReference(source);
  const ExampleSparseMatrix<IT, NT> expected_local = ToEigenReference(coo.local);
  const ExampleSparseMatrix<IT, NT> csr_error = ToEigenReference(csr.local) - expected_local;
  const ExampleSparseMatrix<IT, NT> csc_error = ToEigenReference(csc.local) - expected_local;
  const ExampleSparseMatrix<IT, NT> dcsc_error = ToEigenReference(dcsc.local) - expected_local;
  const bool matches = CheckDistributedReference(copied, reference);
  const int local_status =
      matches && csr_error.norm() == NT{0} && csc_error.norm() == NT{0} && dcsc_error.norm() == NT{0} ? 0 : 1;
  int status = 0;
  MPI_Allreduce(&local_status, &status, 1, MPI_INT, MPI_MAX, grid->World());
  const int64_t global_nnz = coo.GlobalNnz();
  if (grid->Rank() == 0) {
    std::cout << "Grid " << grid->Rows() << " x " << grid->Cols() << ", global nnz " << global_nnz
              << "; distributed COO/CSR/CSC/DCSC Eigen checks " << (status == 0 ? "passed" : "failed")
              << '\n';
  }
  return status;
}

int main(int argc, char** argv)
{
  cxxopts::Options parser(argv[0], "Distributed sparse storage with one submatrix per rank");
  AddMatrixInputOptions(parser);
  cxxopts::OptionAdder flags = parser.add_options();
  const std::shared_ptr<cxxopts::Value> rows = cxxopts::value<int>()->default_value("0");
  const std::shared_ptr<cxxopts::Value> cols = cxxopts::value<int>()->default_value("0");
  flags("grid-rows", "Process rows (0 lets MPI choose)", rows);
  flags("grid-cols", "Process columns (0 lets MPI choose)", cols);
  flags("h,help", "Show help");
  MPI_Init(&argc, &argv);
  int status = 0;
  try {
    const cxxopts::ParseResult arguments = parser.parse(argc, argv);
    if (arguments.count("help")) {
      std::cout << parser.help() << '\n';
    } else {
      const MatrixInputOptions input = ReadMatrixInputOptions(arguments);
      const int grid_rows = arguments["grid-rows"].as<int>();
      const int grid_cols = arguments["grid-cols"].as<int>();
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
