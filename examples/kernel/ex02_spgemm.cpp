#include <chrono>
#include <cstdint>
#include <exception>
#include <type_traits>

#include <fmt/format.h>

#include "SpCraft.h"

using namespace spcraft;

template <class IT, class NT, class OT>
int ExampleRun(const MatrixInputOptions& input)
{
  using Ring = PlusTimesRing<NT>;
  using Clock = std::chrono::steady_clock;
  using Seconds = std::chrono::duration<double>;

  // Use the same square matrix for both operands, with the selected IT/NT/OT.
  const SPCOO coo = ReadMatrixInput<IT, NT, OT>(input.matrix_path);
  const CscMatrix<IT, NT, OT> matrix = coo.ToCsc();

  // Call the tested kernel; reference conversion and checking happen afterward.
  const auto start = Clock::now();
  const SPCOO result = [&] {
    // @profiler-benchmark time_to_solution start
    return OmpHashSpGEMM<Ring>(matrix, matrix);
    // @profiler-benchmark time_to_solution end
  }();

  const auto stop = Clock::now();
  const Seconds elapsed = stop - start;
  const SPCSC result_csc = result.ToCsc();
  constexpr NT kTolerance = std::is_same_v<NT, float> ? NT{1e-6} : NT{1e-13};
  const auto comparison = CompareSpGEMMWithEigen(matrix, matrix, result_csc, kTolerance);

  // Report correctness and explicitly label these illustrative timings.
  if (comparison.matches) {
    constexpr auto kSuccess = "SpGEMM verification passed: matches Eigen (tol = {})\n";
    fmt::print(kSuccess, kTolerance);
  } else {
    constexpr auto kFailure = "SpGEMM verification failed: error = {}, tol = {}\n";
    fmt::print(stderr, kFailure, comparison.error, kTolerance);
  }
  fmt::print("Example timing is not accurate; use benchmarks/ for more accurate timing.\n");
  const double spcraft_seconds = elapsed.count();
  const double eigen_seconds = comparison.multiply_seconds;
  fmt::print("SpCraft: {:.6f} s\nEigen:   {:.6f} s\n", spcraft_seconds, eigen_seconds);
  return comparison.matches ? 0 : 1;
}

int main(int argc, char** argv)
{
  constexpr auto kDescription = "Verify SpCraft OpenMP SpGEMM against Eigen";
  try {
    // @profiler-config argc argv
    const MatrixInput input = ParseMatrixInput(argc, argv, kDescription);
    if (!input.options) return input.exit_code;
    const auto run = [&]<class IT, class NT, class OT>() { return ExampleRun<IT, NT, OT>(*input.options); };
    return DispatchMatrixInputTypes(*input.options, run);
  } catch (const std::exception& error) {
    fmt::print(stderr, "{}\n", error.what());
    return 1;
  }
}
