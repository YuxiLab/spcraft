// Compare normal equations with the numerically safer QR formulation.

#include <Eigen/Dense>
#include <cxxopts.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0], "Fit a polynomial by normal equations and QR");
  options.add_options()("n,points", "Number of sample points",
                        cxxopts::value<int>()->default_value("30"))(
      "d,degree", "Polynomial degree", cxxopts::value<int>()->default_value("10"))("h,help",
                                                                                   "Show usage");
  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0) {
    std::cout << options.help();
    return 0;
  }

  const int points = arguments["points"].as<int>();
  const int degree = arguments["degree"].as<int>();
  if (points <= degree || degree < 0) {
    std::cerr << "require --points > --degree >= 0\n";
    return 1;
  }

  Eigen::MatrixXd design(points, degree + 1);
  Eigen::VectorXd observations(points);
  for (int row = 0; row < points; ++row) {
    const double x = static_cast<double>(row) / static_cast<double>(points - 1);
    double power = 1.0;
    for (int column = 0; column <= degree; ++column) {
      design(row, column) = power;
      power *= x;
    }
    observations(row) = std::exp(x);
  }

  const Eigen::VectorXd normal =
      (design.transpose() * design).ldlt().solve(design.transpose() * observations);
  const Eigen::VectorXd qr = design.colPivHouseholderQr().solve(observations);

  std::cout << std::scientific << std::setprecision(3);
  std::cout << "normal-equation residual: " << (design * normal - observations).norm() << '\n';
  std::cout << "QR residual:              " << (design * qr - observations).norm() << '\n';
  std::cout << "coefficient disagreement: " << (normal - qr).norm() << '\n';
  std::cout << "QR numerical rank: " << design.colPivHouseholderQr().rank() << " / "
            << design.cols() << '\n';
}
