#include "SpCraft.h"
#include "SummaExampleCommon.h"

#include <iostream>

#include <cxxopts.hpp>

using namespace spcraft;

template <class IT, class NT, class OT>
int ExampleRun(const MatrixInputOptions& input, int layers)
{
  const CooMatrix<IT, NT, OT> source = ReadMatrixInput<IT, NT, OT>(input.matrix_path);
  if (source.m != source.n) throw std::invalid_argument("This A*A example needs a square matrix");
  const CommGrid3D grid(MPI_COMM_WORLD, layers);
  const IT extent = source.n / grid.Plane()->Cols() + (source.n % grid.Plane()->Cols() != 0);
  const IT block = std::max(IT{1}, extent);
  int rank = 0;
  MPI_Comm_rank(grid.World(), &rank);

  // A and B need different ownership predicates: layer selection follows inner k.
  const auto owns_a = [&](IT row, IT col) { return grid.AOwner(row / block, col / block) == rank; };
  const auto owns_b = [&](IT row, IT col) { return grid.BOwner(row / block, col / block) == rank; };
  const DistSpCSC<IT, NT, OT> a = PartitionSummaInput(source, grid.Plane(), owns_a);
  const DistSpCSC<IT, NT, OT> b = PartitionSummaInput(source, grid.Plane(), owns_b);
  const DistSpCOO<IT, NT, OT> product = Mpi3DSummaSpGEMM<PlusTimesRing<NT>>(a, b, grid);

  // Only layer zero has the reduced C. Every rank still joins the status reduction.
  int local_status = 0;
  if (grid.Layer() == 0) {
    const ExampleSparseMatrix<IT, NT> reference = ToEigenReference(source);
    const ExampleSparseMatrix<IT, NT> expected = reference * reference;
    local_status = CheckSummaReference(product, expected) ? 0 : 1;
  }
  int status = 0;
  MPI_Allreduce(&local_status, &status, 1, MPI_INT, MPI_MAX, grid.World());
  if (rank == 0) {
    std::cout << "3D SUMMA with " << layers << " layers; layer-zero Eigen check "
              << (status == 0 ? "passed" : "failed") << '\n';
  }
  return status;
}

int main(int argc, char** argv)
{
  cxxopts::Options parser(argv[0], "Layered synchronous 3D SUMMA: C=A*A");
  AddMatrixInputOptions(parser);
  cxxopts::OptionAdder flags = parser.add_options();
  flags("layers", "Layer count; ranks/layers must be square", cxxopts::value<int>());
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
      const int layers = args["layers"].as<int>();
      const auto run = [&]<class IT, class NT, class OT>() { return ExampleRun<IT, NT, OT>(input, layers); };
      status = DispatchMatrixInputTypes(input, run);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  MPI_Finalize();
  return status;
}
