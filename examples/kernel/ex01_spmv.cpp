#include <chrono>
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

  // Load the chosen storage types and convert through the public matrix API.
  const auto coo = ReadMatrixInput<IT, NT, OT>(input.matrix_path);
  const auto matrix = coo.ToCsr();
  DenseVector<IT, NT> x(matrix.n);
  // Keep the input vector reproducible across verification runs.
  constexpr int kSeed = 2026;
  x.Random(kSeed);

  // Call the tested kernel, then independently verify with Eigen.
  const auto start = Clock::now();
  const DenseVector<IT, NT> y = [&] {
    // @profiler-benchmark time_to_solution start
    DenseVector<IT, NT> output(matrix.m);
    // @profiler-region spmv start
    OmpSpMV<Ring>(matrix, x, output);
    // @profiler-region spmv end
    return output;
    // @profiler-benchmark time_to_solution end
  }();
  const auto stop = Clock::now();
  const Seconds elapsed = stop - start;
  constexpr NT kTolerance = std::is_same_v<NT, float> ? NT{1e-6} : NT{1e-13};
  const auto comparison = CompareSpMVWithEigen(matrix, x, y, kTolerance);

  // These timings illustrate the calls; use benchmark drivers for performance.
  if (comparison.matches) {
    constexpr auto kSuccess = "SpMV verification passed: matches Eigen (tol = {})\n";
    fmt::print(kSuccess, kTolerance);
  } else {
    constexpr auto kFailure = "SpMV verification failed: error = {}, tol = {}\n";
    fmt::print(stderr, kFailure, comparison.error, kTolerance);
  }
  fmt::print("Example timing is not accurate; use benchmarks/ for more accurate timing.\n");
  fmt::print("SpCraft: {:.6f} s\n", elapsed.count());
  fmt::print("Eigen:   {:.6f} s\n", comparison.multiply_seconds);
  return comparison.matches ? 0 : 1;
}

int main(int argc, char** argv)
{
  constexpr auto kDescription = "Verify SpCraft OpenMP SpMV against Eigen";
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
