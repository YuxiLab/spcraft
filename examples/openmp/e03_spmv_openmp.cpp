/*******************************************************
 * An example of openmp spmv using csr format.
 * Author: Yuxi Hong
 * Date: 2026-09-01
 * *****************************************************/
// include std library
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
// include third party library
#include <fmt/format.h>
#include <omp.h>
#include <cxxopts.hpp>
// include project header
#include <SpCraft.h>

using namespace spcraft;
using CSR = CsrMatrix<int64_t, double>;
using COO = CooMatrix<int64_t, double>;

int main(int argc, char** argv)
{
  /*****************************************
   * parsing input
   * **************************************/
  const std::string default_filepath = "./dataset/suitsparse/ecology1/ecology1.mtx";
  cxxopts::Options options(argv[0], "Load a Matrix Market file for OpenMP SpMV");
  // clang-format off
  options.add_options()
      ("filepath", "Path to the Matrix Market file", cxxopts::value<std::string>()->default_value(default_filepath))
      ("h,help", "Show usage");
  // clang-format on
  options.parse_positional({"filepath"});
  options.positional_help("[filepath]");
  const auto result = options.parse(argc, argv);
  if (result.count("help") != 0) {
    fmt::print("{}", options.help());
    return 0;
  }
  if (!result.unmatched().empty()) {
    fmt::print(stderr, "Error: expected at most one filepath.\n\n{}", options.help());
    return 1;
  }
  const std::string filepath = result["filepath"].as<std::string>();
  if (!std::filesystem::exists(filepath)) {
    fmt::print(stderr, "Error: matrix file does not exist: {}\n", filepath);
    return 1;
  }

  /*****************************************
   * read coo
   * **************************************/
  CSR A;
  auto Acoo = COO::FromMatrixMarket(filepath);
  fmt::print("nnz {}", Acoo.nnz);
  return 0;
}
