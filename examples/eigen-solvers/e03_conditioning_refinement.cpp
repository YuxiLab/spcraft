// A small residual and a small solution error are different promises.

#include <Eigen/Dense>
#include <cxxopts.hpp>

#include <iomanip>
#include <iostream>

namespace
{

Eigen::MatrixXd Hilbert(int size)
{
  Eigen::MatrixXd matrix(size, size);
  for (int row = 0; row < size; ++row) {
    for (int column = 0; column < size; ++column) {
      matrix(row, column) = 1.0 / static_cast<double>(row + column + 1);
    }
  }
  return matrix;
}

double ConditionNumber(const Eigen::MatrixXd& matrix)
{
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
  const auto singular_values = svd.singularValues();
  return singular_values(0) / singular_values(singular_values.size() - 1);
}

}  // namespace

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0], "Inspect error, residual, and iterative refinement");
  options.add_options()("n,size", "Hilbert matrix dimension",
                        cxxopts::value<int>()->default_value("10"))(
      "r,refinements", "Iterative-refinement steps", cxxopts::value<int>()->default_value("3"))(
      "h,help", "Show usage");
  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0) {
    std::cout << options.help();
    return 0;
  }

  const int size = arguments["size"].as<int>();
  if (size < 2) {
    std::cerr << "--size must be at least 2\n";
    return 1;
  }
  const Eigen::MatrixXd matrix = Hilbert(size);
  const Eigen::VectorXd expected = Eigen::VectorXd::Ones(size);
  const Eigen::VectorXd rhs = matrix * expected;
  const Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> high_precision_matrix =
      matrix.cast<long double>();
  const Eigen::Matrix<long double, Eigen::Dynamic, 1> high_precision_rhs = rhs.cast<long double>();
  const Eigen::Matrix<long double, Eigen::Dynamic, 1> reference =
      high_precision_matrix.partialPivLu().solve(high_precision_rhs);
  const Eigen::PartialPivLU<Eigen::MatrixXd> factorization(matrix);
  Eigen::VectorXd solution = factorization.solve(rhs);

  std::cout << std::scientific << std::setprecision(3);
  std::cout << "estimated cond_2(A): " << ConditionNumber(matrix) << '\n';
  for (int step = 0; step <= arguments["refinements"].as<int>(); ++step) {
    // Compute the residual more accurately than the LU factors and correction solve.
    const Eigen::VectorXd residual =
        (high_precision_rhs - high_precision_matrix * solution.cast<long double>()).cast<double>();
    const long double relative_error =
        (solution.cast<long double>() - reference).norm() / reference.norm();
    std::cout << "step " << step << ": relative residual = " << residual.norm() / rhs.norm()
              << ", relative error = " << relative_error << '\n';
    solution += factorization.solve(residual);
  }
}
