// Triangular solves are the final step in every factorization-based solver.

#include <Eigen/Dense>
#include <cxxopts.hpp>

#include <iostream>

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0], "Solve L y = b and U x = y with Eigen");
  options.add_options()("s,scale", "Scale the right-hand side",
                        cxxopts::value<double>()->default_value("1.0"))("h,help", "Show usage");
  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0) {
    std::cout << options.help();
    return 0;
  }

  const double scale = arguments["scale"].as<double>();
  Eigen::Matrix3d lower;
  lower << 2.0, 0.0, 0.0, -1.0, 3.0, 0.0, 2.0, 1.0, 1.0;
  Eigen::Matrix3d upper;
  upper << 1.0, 2.0, -1.0, 0.0, 4.0, 2.0, 0.0, 0.0, 5.0;
  const Eigen::Vector3d expected = scale * Eigen::Vector3d(1.0, -2.0, 3.0);
  const Eigen::Vector3d rhs = lower * upper * expected;

  // Eigen reads only the declared triangle. No inverse is ever formed.
  const Eigen::Vector3d intermediate = lower.triangularView<Eigen::Lower>().solve(rhs);
  const Eigen::Vector3d solution = upper.triangularView<Eigen::Upper>().solve(intermediate);

  std::cout << "L =\n" << lower << "\n\nU =\n" << upper << "\n\n";
  std::cout << "b = " << rhs.transpose() << '\n';
  std::cout << "x = " << solution.transpose() << '\n';
  std::cout << "relative residual = " << (lower * upper * solution - rhs).norm() / rhs.norm()
            << '\n';
}
