// SVD exposes numerical rank and returns a minimum-norm least-squares solution.

#include <Eigen/Dense>
#include <cxxopts.hpp>

#include <iomanip>
#include <iostream>

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0], "Solve a rank-deficient least-squares problem with SVD");
  options.add_options()("e,epsilon", "Perturb the dependent column",
                        cxxopts::value<double>()->default_value("0"))(
      "t,threshold", "Relative singular-value threshold",
      cxxopts::value<double>()->default_value("1e-12"))("h,help", "Show usage");
  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0) {
    std::cout << options.help();
    return 0;
  }

  Eigen::Matrix<double, 5, 3> matrix;
  matrix << 1, 0, 1, 1, 1, 2, 1, 2, 3, 1, 3, 4, 1, 4, 5;
  matrix(4, 2) += arguments["epsilon"].as<double>();
  const Eigen::Vector<double, 5> rhs(1.0, 2.0, 2.0, 4.0, 5.0);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  svd.setThreshold(arguments["threshold"].as<double>());
  const Eigen::VectorXd solution = svd.solve(rhs);

  std::cout << std::scientific << std::setprecision(4);
  std::cout << "singular values: " << svd.singularValues().transpose() << '\n';
  std::cout << "numerical rank: " << svd.rank() << '\n';
  std::cout << "minimum-norm x: " << solution.transpose() << '\n';
  std::cout << "residual norm: " << (matrix * solution - rhs).norm() << '\n';
}
