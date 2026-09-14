/*******************************************************
 * OpenMP SpMV using CSR storage.
 * Author: Yuxi Hong
 * Date: 2026-09-01
 * *****************************************************/
// Standard library headers.
#include <cstdint>
#include <filesystem>
#include <string>
#include <type_traits>

// Third-party dependencies.
#include <cxxopts.hpp>
#include <fmt/format.h>

// SpCraft and the Eigen conversion helper used for verification.
#include "SpCraft.h"
#include "eigeninterface.h"

using namespace spcraft;

// This example uses 64-bit indices and double-precision values. Change these
// aliases to try another supported type combination.
using IT = std::int64_t;
using NT = double;

using CSR = CsrMatrix<IT, NT>;
using COO = CooMatrix<IT, NT>;
using VEC = DenseVector<IT, NT>;
using Ring = PlusTimesRing<NT>;

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0], "Verify SpCraft OpenMP SpMV against Eigen");

  // clang-format off
  options.positional_help("<matrix.mtx>");
  options.add_options()
  ("matrix", "Matrix Market input file", cxxopts::value<std::string>())
  ("h,help", "Print usage");
  options.parse_positional({"matrix"});
  // clang-format on

  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0 || arguments.count("matrix") == 0) {
    fmt::print("{}\n", options.help());
    return arguments.count("help") != 0 ? 0 : 1;
  }

  const auto filepath = arguments["matrix"].as<std::string>();
  if (!std::filesystem::exists(filepath)) {
    fmt::print(stderr, "Matrix file does not exist: {}\n", filepath);
    return 1;
  }

  COO acoo(filepath);
  CSR acsr = acoo.ToCsr();
  VEC x(acoo.n), y(acoo.m), refy(acoo.m);

  // Use a fixed seed so that verification is repeatable.
  x.Random(2026);
  OmpSpMV<Ring>(acsr, x, y);

  // Compute the same product independently with Eigen.
  ToEigen(refy) = ToEigen(acsr) * ToEigen(x);
  constexpr NT tol = std::is_same_v<NT, float> ? 1e-6f : 1e-13;
  const bool match = ToEigen(y).isApprox(ToEigen(refy), tol);
  if (match) {
    fmt::print("SpMV verification passed: output matches Eigen reference (tol = {})\n", tol);
  } else {
    fmt::print(stderr, "SpMV verification failed: output differs from Eigen reference\n");
  }
  return match ? 0 : 1;
}
