// Show why a tiny first pivot makes elimination without row exchanges unsafe.

#include <Eigen/Dense>
#include <cxxopts.hpp>

#include <cmath>
#include <iostream>

namespace
{

Eigen::Vector2d SolveWithoutPivoting(const Eigen::Matrix2d& matrix, const Eigen::Vector2d& rhs)
{
  Eigen::Matrix2d upper = matrix;
  Eigen::Vector2d transformed_rhs = rhs;
  const double multiplier = upper(1, 0) / upper(0, 0);
  upper.row(1) -= multiplier * upper.row(0);
  transformed_rhs(1) -= multiplier * transformed_rhs(0);
  return upper.triangularView<Eigen::Upper>().solve(transformed_rhs);
}

}  // namespace

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0], "Compare unpivoted elimination and Eigen PartialPivLU");
  options.add_options()("p,pivot", "The unsafe leading pivot",
                        cxxopts::value<double>()->default_value("1e-20"))("h,help", "Show usage");
  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0) {
    std::cout << options.help();
    return 0;
  }

  Eigen::Matrix2d matrix;
  matrix << arguments["pivot"].as<double>(), 1.0, 1.0, 1.0;
  const Eigen::Vector2d expected(1.0, 1.0);
  const Eigen::Vector2d rhs = matrix * expected;
  const Eigen::Vector2d no_pivot = SolveWithoutPivoting(matrix, rhs);
  const Eigen::Vector2d pivoted = matrix.partialPivLu().solve(rhs);

  std::cout << "A =\n" << matrix << "\n\n";
  std::cout << "without pivoting: " << no_pivot.transpose() << '\n';
  std::cout << "with PartialPivLU: " << pivoted.transpose() << '\n';
  std::cout << "unpivoted error: " << (no_pivot - expected).norm() << '\n';
  std::cout << "pivoted error:   " << (pivoted - expected).norm() << '\n';
}
