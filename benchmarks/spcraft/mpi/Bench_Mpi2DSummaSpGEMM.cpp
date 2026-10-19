#include "SpCraft.h"
#include "utils/MatrixInput.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/SparseCore>
#include <cxxopts.hpp>

using namespace spcraft;

namespace
{

using InputCOO = CooMatrix<int64_t, double, int64_t>;
using DistributedCSC = DistSpCSC<int64_t, double, int64_t>;
using DistributedCOO = DistSpCOO<int64_t, double, int64_t>;
using ReferenceMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t>;

DistributedCSC DistributeInput(const InputCOO& source, std::shared_ptr<const CommGrid2D> grid)
{
  DistributedCOO selected(grid, source.m, source.n);
  int64_t count = 0;
  for (int64_t p = 0; p < source.nnz; ++p) {
    const auto& entry = source.entries[p];
    if (selected.Owns(entry.row, entry.col)) ++count;
  }
  selected.Allocate(count);
  int64_t position = 0;
  for (int64_t p = 0; p < source.nnz; ++p) {
    const auto& entry = source.entries[p];
    if (!selected.Owns(entry.row, entry.col)) continue;
    selected.local.entries[position++] = {selected.LocalRow(entry.row), selected.LocalCol(entry.col),
                                          entry.val};
  }
  CscMatrix<int64_t, double, int64_t> local = selected.local.ToCsc();
  DistributedCSC result(grid, std::move(local), source.m, source.n);
  return result;
}

ReferenceMatrix ToReference(const InputCOO& source)
{
  std::vector<Eigen::Triplet<double, int64_t>> entries;
  entries.reserve(source.nnz);
  for (int64_t p = 0; p < source.nnz; ++p) {
    const auto& entry = source.entries[p];
    entries.emplace_back(entry.row, entry.col, entry.val);
  }
  ReferenceMatrix result(source.m, source.n);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

int RunSummaBenchmark(const std::string& a_path, const std::string& b_path)
{
  OMP_SET_NUM_THREADS(1);
  for (const std::string& path : {a_path, b_path}) {
    if (!std::filesystem::is_regular_file(path)) {
      throw std::invalid_argument("Matrix input is not a regular file: " + path);
    }
  }
  const InputCOO source_a = ReadMatrixInput<int64_t, double, int64_t>(a_path);
  std::optional<InputCOO> separate_source_b;
  if (b_path != a_path) separate_source_b = ReadMatrixInput<int64_t, double, int64_t>(b_path);
  const InputCOO& source_b = separate_source_b ? *separate_source_b : source_a;
  if (source_a.n != source_b.m) throw std::invalid_argument("A columns must match B rows");

  const std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const DistributedCSC a = DistributeInput(source_a, grid);
  std::optional<DistributedCSC> separate_b;
  if (separate_source_b) separate_b = DistributeInput(source_b, grid);
  const DistributedCSC& b = separate_b ? *separate_b : a;
  const DistributedCOO product = [&] {
    // @profiler-benchmark time_to_solution start
    return Mpi2DSummaSpGEMM<PlusTimesRing<double>>(a, b);
    // @profiler-benchmark time_to_solution end
  }();

  const ReferenceMatrix expected = ToReference(source_a) * ToReference(source_b);
  std::vector<Eigen::Triplet<double, int64_t>> entries;
  for (int64_t col = 0; col < expected.outerSize(); ++col) {
    for (ReferenceMatrix::InnerIterator entry(expected, col); entry; ++entry) {
      if (product.Owns(entry.row(), col)) {
        entries.emplace_back(product.LocalRow(entry.row()), product.LocalCol(col), entry.value());
      }
    }
  }
  ReferenceMatrix local_expected(product.local.m, product.local.n);
  local_expected.setFromTriplets(entries.begin(), entries.end());
  const ReferenceMatrix error = ToReference(product.local) - local_expected;
  // Allow roundoff from different accumulation orders in the two products.
  constexpr double kTolerance = 100 * std::numeric_limits<double>::epsilon();
  const bool matches = product.m == source_a.m && product.n == source_b.n &&
                       error.norm() <= kTolerance * std::max(1.0, local_expected.norm());
  const int failed = matches ? 0 : 1;
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, grid->World());
  if (grid->Rank() == 0) {
    std::cout << "2D SUMMA CSC check " << (any_failed == 0 ? "passed" : "failed") << " (" << source_a.m << 'x'
              << source_a.n << " * " << source_b.m << 'x' << source_b.n << ")\n";
  }
  return any_failed;
}

}  // namespace

int main(int argc, char** argv)
{
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int status = 0;
  try {
    // @profiler-config argc argv
    if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI needs FUNNELED threading");
    cxxopts::Options parser(argv[0], "2D SUMMA CSC: C=A*B (B defaults to A)");
    cxxopts::OptionAdder flags = parser.add_options();
    flags("a,matrix-a", "Matrix Market input A (required)", cxxopts::value<std::string>());
    flags("b,matrix-b", "Matrix Market input B (default: A)", cxxopts::value<std::string>());
    flags("h,help", "Show help");
    parser.parse_positional({"matrix-a", "matrix-b"});
    parser.positional_help("A.mtx [B.mtx]");
    parser.show_positional_help();
    const cxxopts::ParseResult args = parser.parse(argc, argv);
    if (args.count("help")) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0) std::cout << parser.help() << '\n';
    } else {
      if (args.count("matrix-a") == 0) throw std::invalid_argument("Missing --matrix-a input file");
      const std::string a_path = args["matrix-a"].as<std::string>();
      const std::string b_path = args.count("matrix-b") ? args["matrix-b"].as<std::string>() : a_path;
      status = RunSummaBenchmark(a_path, b_path);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    status = 1;
  }
  MPI_Finalize();
  return status;
}
