/*******************************************************
 * OpenMP SpGEMM using CSC storage.
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

using CSC = CscMatrix<IT, NT>;
using COO = CooMatrix<IT, NT>;
using Ring = PlusTimesRing<NT>;

int main(int argc, char** argv)
{
  const auto input = ParseMatrixInput(argc, argv, "Verify SpCraft OpenMP SpGEMM against Eigen");
  if (!input.path) return input.exit_code;

  COO acoo = ReadMatrixInput(*input.path);
  CSC acsc = acoo.ToCsc();
  COO bcoo = ReadMatrixInput(*input.path);
  CSC bcsc = bcoo.ToCsc();
  const auto result = OmpHashSpGEMM<Ring, IT, NT, IT>(acsc, bcsc);
  const auto result_csc = result.ToCsc();
  constexpr NT tol = std::is_same_v<NT, float> ? 1e-6f : 1e-13;
  if (result_csc.m != acsc.m || result_csc.n != bcsc.n) {
    fmt::print(stderr, "SpGEMM verification failed: output shape is {} x {}, expected {} x {}\n",
               result_csc.m, result_csc.n, acsc.m, bcsc.n);
    return 1;
  }
  const auto comparison = CompareSpGEMMWithEigen(acsc, bcsc, result_csc, tol);
  if (comparison.matches) {
    fmt::print("SpGEMM verification passed: output matches Eigen reference (tol = {})\n", tol);
  } else {
    fmt::print(stderr,
               "SpGEMM verification failed: output differs from Eigen reference "
               "(error = {}, tol = {})\n",
               comparison.error, tol);
  }
  return comparison.matches ? 0 : 1;
}
