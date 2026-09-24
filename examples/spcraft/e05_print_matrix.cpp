// Print small matrices and select a subblock without changing the source.
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cxxopts.hpp>
#include "SpCraft.h"

int main(int argc, char** argv)
{
  try {
    cxxopts::Options options(argv[0], "Print SpCraft and Eigen matrix subblocks");
    options.add_options()("row", "First row (zero-based)",
                          cxxopts::value<std::int64_t>()->default_value("1"))(
        "col", "First column (zero-based)", cxxopts::value<std::int64_t>()->default_value("1"))(
        "rows", "Number of rows", cxxopts::value<std::int64_t>()->default_value("2"))(
        "cols", "Number of columns", cxxopts::value<std::int64_t>()->default_value("3"))(
        "h,help", "Show usage");
    const auto args = options.parse(argc, argv);
    if (args.count("help")) {
      std::cout << options.help() << '\n';
      return 0;
    }
    const auto row = args["row"].as<std::int64_t>();
    const auto col = args["col"].as<std::int64_t>();
    const auto rows = args["rows"].as<std::int64_t>();
    const auto cols = args["cols"].as<std::int64_t>();

    // An explicitly stored zero looks different from an absent sparse entry.
    spcraft::CooMatrix<int, double> coo;
    coo.Allocate(5, 4, 5);
    coo.entries[0] = {0, 4, 9};
    coo.entries[1] = {1, 1, 0};
    coo.entries[2] = {1, 2, 5};
    coo.entries[3] = {2, 3, 8};
    coo.entries[4] = {3, 0, 7};

    // The short overload starts at (0, 0). The requested size is clipped to fit.
    std::cout << "SpCraft: '.' means no stored entry; '0' is a stored zero.\n";
    spcraft::PrintMatrix(coo, coo.m, coo.n, "Full COO matrix");

    // The long overload takes the starting row/column, then the block dimensions.
    // The row and column labels keep their original matrix indices.
    const auto csr = coo.ToCsr();
    const auto csc = coo.ToCsc();
    spcraft::PrintMatrix(coo, row, col, rows, cols, "COO subblock");
    spcraft::PrintMatrix(csr, row, col, rows, cols, "CSR subblock");
    spcraft::PrintMatrix(csc, row, col, rows, cols, "CSC subblock");

    // Eigen blocks use Eigen's own stream output, where absent entries print as zero.
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(coo.m, coo.n);
    for (int p = 0; p < coo.nnz; ++p) {
      const auto& entry = coo.entries[p];
      dense(entry.row, entry.col) = entry.val;
    }
    const Eigen::SparseMatrix<double> sparse = dense.sparseView();
    std::cout << "\nEigen native formatting:\n";
    spcraft::PrintMatrix(dense, row, col, rows, cols, "Eigen dense subblock");
    spcraft::PrintMatrix(sparse, row, col, rows, cols, "Eigen sparse subblock");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
