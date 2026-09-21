// Cholesky is the specialized direct solver for symmetric positive-definite matrices.

#include <Eigen/Dense>
#include <cxxopts.hpp>

#include <chrono>
#include <iostream>
#include <random>

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0], "Compare dense Cholesky and pivoted LU");
  options.add_options()("n,size", "Matrix dimension", cxxopts::value<int>()->default_value("600"))(
      "s,seed", "Random seed", cxxopts::value<unsigned>()->default_value("7"))("h,help",
                                                                               "Show usage");
  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0) {
    std::cout << options.help();
    return 0;
  }

  const int size = arguments["size"].as<int>();
  if (size < 1) {
    std::cerr << "--size must be positive\n";
    return 1;
  }
  std::mt19937 generator(arguments["seed"].as<unsigned>());
  std::normal_distribution<double> distribution;
  Eigen::MatrixXd random(size, size);
  for (double& value : random.reshaped()) value = distribution(generator);
  const Eigen::MatrixXd matrix =
      random.transpose() * random + size * Eigen::MatrixXd::Identity(size, size);
  const Eigen::VectorXd expected = Eigen::VectorXd::Ones(size);
  const Eigen::VectorXd rhs = matrix * expected;

  const auto start_llt = std::chrono::steady_clock::now();
  const Eigen::VectorXd cholesky_solution = matrix.llt().solve(rhs);
  const auto stop_llt = std::chrono::steady_clock::now();
  const auto start_lu = std::chrono::steady_clock::now();
  const Eigen::VectorXd lu_solution = matrix.partialPivLu().solve(rhs);
  const auto stop_lu = std::chrono::steady_clock::now();

  const auto milliseconds = [](auto begin, auto end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  };
  std::cout << "LLT time: " << milliseconds(start_llt, stop_llt)
            << " ms, error: " << (cholesky_solution - expected).norm() / expected.norm() << '\n';
  std::cout << "LU time:  " << milliseconds(start_lu, stop_lu)
            << " ms, error: " << (lu_solution - expected).norm() / expected.norm() << '\n';
}
