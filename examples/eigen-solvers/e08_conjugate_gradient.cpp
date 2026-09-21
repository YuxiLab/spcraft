// A transparent conjugate-gradient loop using Eigen sparse matrix-vector products.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cxxopts.hpp>

#include <cmath>
#include <iomanip>
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
  cxxopts::Options options(argv[0], "Run CG and print a convergence history as CSV");
  options.add_options()("g,grid", "Grid points per dimension",
                        cxxopts::value<int>()->default_value("32"))(
      "m,max-iterations", "Maximum iterations", cxxopts::value<int>()->default_value("500"))(
      "t,tolerance", "Relative residual tolerance",
      cxxopts::value<double>()->default_value("1e-10"))("h,help", "Show usage");
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
  const auto matrix = Laplacian2D(grid);
  const Eigen::VectorXd exact = Eigen::VectorXd::Ones(matrix.rows());
  const Eigen::VectorXd rhs = matrix * exact;
  Eigen::VectorXd solution = Eigen::VectorXd::Zero(matrix.rows());
  Eigen::VectorXd residual = rhs;
  Eigen::VectorXd direction = residual;
  double residual_squared = residual.squaredNorm();
  const double initial_norm = std::sqrt(residual_squared);

  std::cout << "iteration,relative_residual\n0,1\n" << std::scientific;
  int iteration = 0;
  for (; iteration < arguments["max-iterations"].as<int>(); ++iteration) {
    const Eigen::VectorXd image = matrix * direction;
    const double step = residual_squared / direction.dot(image);
    solution += step * direction;
    residual -= step * image;
    const double next_residual_squared = residual.squaredNorm();
    const double relative_residual = std::sqrt(next_residual_squared) / initial_norm;
    std::cout << iteration + 1 << ',' << relative_residual << '\n';
    if (relative_residual <= arguments["tolerance"].as<double>()) break;
    direction = residual + (next_residual_squared / residual_squared) * direction;
    residual_squared = next_residual_squared;
  }
  std::cerr << "iterations: " << iteration + 1
            << ", relative solution error: " << (solution - exact).norm() / exact.norm() << '\n';
}
