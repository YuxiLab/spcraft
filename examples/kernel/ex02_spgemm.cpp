#include <chrono>
#include <cstdint>
#include <numeric>
#include <string>
#include <type_traits>

#include <fmt/format.h>

#include "SpCraft.h"
#include "eigeninterface.h"

namespace spcraft
{
namespace
{
int ExampleRun(const std::string& matrix_path)
{
  // Match the compiled Eigen reference's 64-bit indices and double-precision values.
  using IT = std::int64_t;
  using NT = double;
  using OT = IT;
  using COO = spcraft::CooMatrix<IT, NT, OT>;
  using CSC = spcraft::CscMatrix<IT, NT, OT>;
  using Ring = spcraft::PlusTimesRing<NT>;
  using Clock = std::chrono::steady_clock;
  using Seconds = std::chrono::duration<double>;

  // Read both operands in COO form, then convert to CSC for column-oriented multiplication.
  const COO lhs_coo = spcraft::ReadMatrixInput(matrix_path);
  const CSC lhs = lhs_coo.ToCsc();
  const COO rhs_coo = spcraft::ReadMatrixInput(matrix_path);
  const CSC rhs = rhs_coo.ToCsc();

  // Time only SpCraft's multiplication; conversion and verification happen afterward.
  const auto start = Clock::now();
  const auto result = spcraft::OmpHashSpGEMM<Ring, IT, NT, OT>(lhs, rhs);
  const auto stop = Clock::now();
  const Seconds elapsed = stop - start;
  const double seconds = elapsed.count();

  // Count one multiply and one add per scalar product for comparable throughput figures.
  constexpr double kFlopsPerProduct = 2.0;
  const auto products = spcraft::HashSpGEMMCscEstimateFlops(lhs, rhs);
  const IT scalar_products = std::accumulate(products.begin(), products.end(), IT{0});
  const double flops = kFlopsPerProduct * static_cast<double>(scalar_products);

  // Verify the output shape before comparing values against the independent Eigen result.
  const auto result_csc = result.ToCsc();
  constexpr NT kTolerance = std::is_same_v<NT, float> ? 1e-6f : 1e-13;
  if (result_csc.m != lhs.m || result_csc.n != rhs.n) {
    constexpr auto kShapeError =
        "SpGEMM verification failed: output shape is {} x {}, expected {} x {}\n";
    fmt::print(stderr, kShapeError, result_csc.m, result_csc.n, lhs.m, rhs.n);
    return 1;
  }
  const auto comparison = spcraft::CompareSpGEMMWithEigen(lhs, rhs, result_csc, kTolerance);

  // Report verification and both multiplication rates using the same FLOP count.
  if (comparison.matches) {
    constexpr auto kSuccess =
        "SpGEMM verification passed: output matches Eigen reference (tol = {})\n";
    fmt::print(kSuccess, kTolerance);
    constexpr double kFlopsPerGflop = 1e9;
    const double spcraft_gflops = flops / seconds / kFlopsPerGflop;
    const double eigen_seconds = comparison.multiply_seconds;
    const double eigen_gflops = flops / eigen_seconds / kFlopsPerGflop;
    fmt::print("FLOPs: {:.0f}\n", flops);
    fmt::print("SpCraft: {:.6f} s, {:.3f} GFLOPS\n", seconds, spcraft_gflops);
    fmt::print("Eigen:   {:.6f} s, {:.3f} GFLOPS\n", eigen_seconds, eigen_gflops);
  } else {
    constexpr auto kValueError =
        "SpGEMM verification failed: output differs from Eigen reference (error = {}, tol = {})\n";
    fmt::print(stderr, kValueError, comparison.error, kTolerance);
  }
  return comparison.matches ? 0 : 1;
}
}  // namespace
}  // namespace spcraft

int main(int argc, char** argv)
{
  // Parse the matrix path, leaving help and input errors to the shared input utility.
  constexpr auto kDescription = "Verify SpCraft OpenMP SpGEMM against Eigen";
  const auto input = spcraft::ParseMatrixInput(argc, argv, kDescription);
  if (!input.path) return input.exit_code;

  // Run the complete example and return its verification status.
  return spcraft::ExampleRun(*input.path);
}
