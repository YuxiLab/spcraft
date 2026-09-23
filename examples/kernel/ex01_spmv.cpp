/*******************************************************
 * OpenMP SpMV using CSR storage.
 * Author: Yuxi Hong
 * Date: 2026-09-01
 * *****************************************************/
// Standard library headers.
#include <cstdint>
#include <type_traits>

// Third-party dependencies.
#include <fmt/format.h>

// SpCraft and the Eigen conversion helper used for verification.
#include "SpCraft.h"
#include "eigeninterface.h"

using namespace spcraft;

// The compiled Eigen reference helper uses 64-bit indices and double-precision values.
using IT = std::int64_t;
using NT = double;

using CSR = CsrMatrix<IT, NT>;
using COO = CooMatrix<IT, NT>;
using VEC = DenseVector<IT, NT>;
using Ring = PlusTimesRing<NT>;

int main(int argc, char** argv)
{
  const auto input = ParseMatrixInput(argc, argv, "Verify SpCraft OpenMP SpMV against Eigen");
  if (!input.path) return input.exit_code;

  COO acoo = ReadMatrixInput(*input.path);
  CSR acsr = acoo.ToCsr();
  VEC x(acoo.n), y(acoo.m);

  // Use a fixed seed so that verification is repeatable.
  x.Random(2026);
  OmpSpMV<Ring>(acsr, x, y);

  constexpr NT tol = std::is_same_v<NT, float> ? 1e-6f : 1e-13;
  const auto comparison = CompareSpMVWithEigen(acsr, x, y, tol);
  if (comparison.matches) {
    fmt::print("SpMV verification passed: output matches Eigen reference (tol = {})\n", tol);
  } else {
    fmt::print(stderr,
               "SpMV verification failed: output differs from Eigen reference "
               "(error = {}, tol = {})\n",
               comparison.error, tol);
  }
  return comparison.matches ? 0 : 1;
}
