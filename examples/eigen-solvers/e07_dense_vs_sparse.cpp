// The same 2D Laplacian solved through dense and sparse direct factorizations.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cxxopts.hpp>

#include <chrono>
#include <iostream>
#include <vector>

namespace
{

Eigen::SparseMatrix<double> Laplacian2D(int grid)
{
  const int size = grid * grid;
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(5 * size);
  for (int row = 0; row < grid; ++row) {
    for (int column = 0; column < grid; ++column) {
      const int point = row * grid + column;
      entries.emplace_back(point, point, 4.0);
      if (row > 0) entries.emplace_back(point, point - grid, -1.0);
      if (row + 1 < grid) entries.emplace_back(point, point + grid, -1.0);
      if (column > 0) entries.emplace_back(point, point - 1, -1.0);
      if (column + 1 < grid) entries.emplace_back(point, point + 1, -1.0);
    }
  }
  Eigen::SparseMatrix<double> matrix(size, size);
  matrix.setFromTriplets(entries.begin(), entries.end());
  return matrix;
}

}  // namespace

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0], "Compare dense and sparse direct solution of a grid problem");
  options.add_options()("g,grid", "Grid points per dimension",
                        cxxopts::value<int>()->default_value("35"))("h,help", "Show usage");
  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0) {
    std::cout << options.help();
    return 0;
  }

  const int grid = arguments["grid"].as<int>();
  if (grid < 2) {
    std::cerr << "--grid must be at least 2\n";
    return 1;
  }
  const Eigen::SparseMatrix<double> sparse = Laplacian2D(grid);
  const Eigen::MatrixXd dense(sparse);
  const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(sparse.rows());

  const auto sparse_start = std::chrono::steady_clock::now();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> sparse_solver(sparse);
  const Eigen::VectorXd sparse_solution = sparse_solver.solve(rhs);
  const auto sparse_stop = std::chrono::steady_clock::now();
  const auto dense_start = std::chrono::steady_clock::now();
  const Eigen::VectorXd dense_solution = dense.ldlt().solve(rhs);
  const auto dense_stop = std::chrono::steady_clock::now();

  const auto milliseconds = [](auto begin, auto end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  };
  const auto dense_bytes = static_cast<double>(dense.size() * sizeof(double));
  const auto sparse_bytes =
      static_cast<double>(sparse.nonZeros() * (sizeof(double) + sizeof(Eigen::Index)) +
                          (sparse.outerSize() + 1) * sizeof(Eigen::Index));
  std::cout << "unknowns: " << sparse.rows() << ", nonzeros: " << sparse.nonZeros() << '\n';
  std::cout << "estimated input storage: dense " << dense_bytes / 1.0e6 << " MB, sparse "
            << sparse_bytes / 1.0e6 << " MB\n";
  std::cout << "dense LDLT: " << milliseconds(dense_start, dense_stop) << " ms\n";
  std::cout << "sparse LDLT: " << milliseconds(sparse_start, sparse_stop) << " ms\n";
  std::cout << "relative disagreement: "
            << (dense_solution - sparse_solution).norm() / dense_solution.norm() << '\n';
}
