#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "MatrixGenerator.h"
#include "SpCraft.h"

namespace
{

using Index = std::int32_t;
using Offset = std::int32_t;

template <class NT>
using Matrix = spcraft::CsrMatrix<Index, NT, Offset>;

template <class NT>
using Vector = spcraft::DenseVector<Index, NT>;

std::vector<int> parse_threads(const std::string& thread_str)
{
  std::vector<int> threads;
  std::stringstream ss(thread_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    std::stringstream token_ss(token);
    int t;
    while (token_ss >> t) {
      if (t > 0) {
        threads.push_back(t);
      }
    }
  }
  if (threads.empty()) {
    int max_t = omp_get_max_threads();
    for (int t = 1; t <= max_t; t *= 2) {
      threads.push_back(t);
    }
    if (threads.empty() || threads.back() != max_t) {
      threads.push_back(max_t);
    }
  }
  return threads;
}

template <class NT>
void run_spmv_benchmarks(const Matrix<NT>& A, const std::string& matrix_name,
                         const std::vector<int>& thread_counts, double min_time,
                         int repetitions, std::uint64_t seed)
{
  // These buffers must outlive benchmark registration: Google Benchmark runs
  // the callbacks after this function returns.
  auto x = std::make_shared<Vector<NT>>(A.n);
  auto spcraft_y = std::make_shared<Vector<NT>>(A.m);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (Index i = 0; i < A.n; ++i) {
    (*x)[i] = static_cast<NT>(dist(rng));
  }

  // Warmup run
  omp_set_num_threads(1);
  spcraft::spmv_openmp<spcraft::PlusTimesRing<NT>>(A, *x, *spcraft_y);

  const std::string type_name = std::is_same_v<NT, float> ? "FP32" : "FP64";
  const std::uint64_t useful_bytes =
      static_cast<std::uint64_t>(A.nnz) * (sizeof(NT) * 2 + sizeof(Index)) +
      static_cast<std::uint64_t>(A.m + 1) * sizeof(Offset) +
      static_cast<std::uint64_t>(A.m) * sizeof(NT);

#ifdef SPCRAFT_USE_MKL
  auto mkl_y = std::make_shared<Vector<NT>>(A.m);
#endif

  for (int threads : thread_counts) {
    const std::string spcraft_name = "SpMV/SPCraft/" + matrix_name + "/" + type_name +
                                     "/threads:" + std::to_string(threads);

    auto* spcraft_benchmark = benchmark::RegisterBenchmark(
        spcraft_name.c_str(),
        [&A, x, spcraft_y, threads, useful_bytes](benchmark::State& state) {
          omp_set_num_threads(threads);
          for (auto _ : state) {
            spcraft::spmv_openmp<spcraft::PlusTimesRing<NT>>(A, *x, *spcraft_y);
            benchmark::DoNotOptimize(spcraft_y->data());
            benchmark::ClobberMemory();
          }

          const double total_flops = 2.0 * static_cast<double>(A.nnz) * state.iterations();
          state.SetItemsProcessed(state.iterations() * A.nnz);
          state.SetBytesProcessed(state.iterations() * useful_bytes);
          state.counters["GFLOPS"] =
              benchmark::Counter(total_flops, benchmark::Counter::kIsRate);
          state.counters["NNZ"] = static_cast<double>(A.nnz);
          state.counters["Rows"] = static_cast<double>(A.m);
          state.counters["Cols"] = static_cast<double>(A.n);
          state.counters["Threads"] = static_cast<double>(threads);
        });

    spcraft_benchmark->MinTime(min_time);
    spcraft_benchmark->Repetitions(repetitions);
    if (repetitions > 1) {
      spcraft_benchmark->ReportAggregatesOnly(true);
    }
    spcraft_benchmark->UseRealTime();

#ifdef SPCRAFT_USE_MKL
    // oneMKL's inspector partitions work using the active thread count. Keep a
    // separately optimized handle for every benchmark configuration; changing
    // mkl_set_num_threads after optimization does not repartition the handle.
    omp_set_num_threads(threads);
    mkl_set_dynamic(0);
    mkl_set_num_threads(threads);
    auto mkl_matrix = std::make_shared<spcraft::MklCsrSpmv<Index, NT, Offset>>(A, 10000);
    mkl_matrix->Multiply(x->data(), mkl_y->data());

    double max_reference = 0.0;
    double max_error = 0.0;
    for (Index row = 0; row < A.m; ++row) {
      max_reference =
          std::max(max_reference, std::abs(static_cast<double>((*spcraft_y)[row])));
      max_error = std::max(max_error,
                           std::abs(static_cast<double>((*spcraft_y)[row] - (*mkl_y)[row])));
    }
    const double tolerance =
        (std::is_same_v<NT, float> ? 1.0e-4 : 1.0e-11) * std::max(1.0, max_reference);
    if (max_error > tolerance) {
      throw std::runtime_error("oneMKL result differs from SPCraft for " + matrix_name +
                               ": max error " + std::to_string(max_error) + ", tolerance " +
                               std::to_string(tolerance));
    }

    const std::string mkl_name =
        "SpMV/MKL/" + matrix_name + "/" + type_name + "/threads:" + std::to_string(threads);
    auto* mkl_benchmark = benchmark::RegisterBenchmark(
        mkl_name.c_str(),
        [mkl_matrix, x, mkl_y, threads, useful_bytes, &A](benchmark::State& state) {
          omp_set_num_threads(threads);
          mkl_set_dynamic(0);
          mkl_set_num_threads(threads);
          for (auto _ : state) {
            mkl_matrix->Multiply(x->data(), mkl_y->data());
            benchmark::DoNotOptimize(mkl_y->data());
            benchmark::ClobberMemory();
          }

          const double total_flops = 2.0 * static_cast<double>(A.nnz) * state.iterations();
          state.SetItemsProcessed(state.iterations() * A.nnz);
          state.SetBytesProcessed(state.iterations() * useful_bytes);
          state.counters["GFLOPS"] =
              benchmark::Counter(total_flops, benchmark::Counter::kIsRate);
          state.counters["NNZ"] = static_cast<double>(A.nnz);
          state.counters["Rows"] = static_cast<double>(A.m);
          state.counters["Cols"] = static_cast<double>(A.n);
          state.counters["Threads"] = static_cast<double>(threads);
        });
    mkl_benchmark->MinTime(min_time);
    mkl_benchmark->Repetitions(repetitions);
    if (repetitions > 1) {
      mkl_benchmark->ReportAggregatesOnly(true);
    }
    mkl_benchmark->UseRealTime();
#endif
  }
}

}  // namespace

int main(int argc, char** argv)
{
  cxxopts::Options options("spmv_benchmark",
                           "SPCraft SpMV Google Benchmark Driver with OpenMP parallelization");

  options.allow_unrecognised_options();

  // clang-format off
  options.add_options()
      ("m,matrix", "Path to Matrix Market (.mtx) file", cxxopts::value<std::string>())
      ("v,vertices", "Number of vertices for synthetic ER graph", cxxopts::value<Index>()->default_value("50000"))
      ("d,degree", "Average degree for synthetic ER graph", cxxopts::value<double>()->default_value("16.0"))
      ("p,precision", "Precision: 'float', 'double', or 'both'", cxxopts::value<std::string>()->default_value("double"))
      ("t,threads", "Comma-separated thread counts (e.g. '1,2,4,8')", cxxopts::value<std::string>()->default_value(""))
      ("s,seed", "Random generator seed", cxxopts::value<std::uint64_t>()->default_value("42"))
      ("min-time", "Minimum benchmark runtime in seconds", cxxopts::value<double>()->default_value("0.5"))
      ("r,repetitions", "Benchmark repetitions per configuration", cxxopts::value<int>()->default_value("3"))
      ("h,help", "Print help message");
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << "\n";
    std::cout << "\nGoogle Benchmark options can also be passed (e.g., --benchmark_filter, "
                 "--benchmark_format=json, --benchmark_out=res.json).\n";
    return 0;
  }

  // Only forward arguments not consumed by cxxopts. Passing SPCraft's custom
  // options to Google Benchmark makes it reject otherwise valid invocations.
  std::vector<std::string> benchmark_arg_storage{argv[0]};
  for (const auto& argument : result.unmatched()) {
    benchmark_arg_storage.push_back(argument);
  }
  std::vector<char*> benchmark_args;
  benchmark_args.reserve(benchmark_arg_storage.size());
  for (auto& argument : benchmark_arg_storage) {
    benchmark_args.push_back(argument.data());
  }
  int benchmark_argc = static_cast<int>(benchmark_args.size());
  benchmark::Initialize(&benchmark_argc, benchmark_args.data());
  if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_args.data())) {
    return 2;
  }

  omp_set_dynamic(0);

  const std::string matrix_path =
      result.count("matrix") ? result["matrix"].as<std::string>() : "";
  const Index vertices = result["vertices"].as<Index>();
  const double degree = result["degree"].as<double>();
  const std::string precision = result["precision"].as<std::string>();
  const std::string thread_arg = result["threads"].as<std::string>();
  const std::uint64_t seed = result["seed"].as<std::uint64_t>();
  const double min_time = result["min-time"].as<double>();
  const int repetitions = result["repetitions"].as<int>();

  std::vector<int> thread_counts = parse_threads(thread_arg);

  std::cout << "====================================================\n";
  std::cout << "           SPCraft SpMV Google Benchmark           \n";
  std::cout << "====================================================\n";
  std::cout << "Threads to test : [";
  for (std::size_t i = 0; i < thread_counts.size(); ++i) {
    std::cout << thread_counts[i] << (i + 1 < thread_counts.size() ? ", " : "");
  }
  std::cout << "]\n";
  std::cout << "Precision       : " << precision << "\n";
  std::cout << "Repetitions     : " << repetitions << "\n";
  std::cout << "Min time / test : " << std::fixed << std::setprecision(2) << min_time << "s\n";
#ifdef SPCRAFT_USE_MKL
  MKLVersion mkl_version;
  mkl_get_version(&mkl_version);
  std::cout << "oneMKL          : " << mkl_version.MajorVersion << '.'
            << mkl_version.MinorVersion << '.' << mkl_version.UpdateVersion
            << " (GNU OpenMP, LP64)\n";
#else
  std::cout << "oneMKL          : disabled\n";
#endif

  // Keep loaded matrices in scope for benchmark execution
  std::unique_ptr<Matrix<float>> f32_matrix;
  std::unique_ptr<Matrix<double>> f64_matrix;
  std::string matrix_name;

  if (!matrix_path.empty()) {
    matrix_name = matrix_path.substr(matrix_path.find_last_of("/\\") + 1);
    std::cout << "Loading Matrix Market file: " << matrix_path << "\n";

    if (precision == "float" || precision == "both") {
      auto coo = spcraft::CooMatrix<Index, float, Offset>::FromMatrixMarket(matrix_path);
      f32_matrix = std::make_unique<Matrix<float>>(coo.ToCsr());
      std::cout << "  [FP32] Rows: " << f32_matrix->m << ", Cols: " << f32_matrix->n
                << ", NNZ: " << f32_matrix->nnz << "\n";
    }
    if (precision == "double" || precision == "both") {
      auto coo = spcraft::CooMatrix<Index, double, Offset>::FromMatrixMarket(matrix_path);
      f64_matrix = std::make_unique<Matrix<double>>(coo.ToCsr());
      std::cout << "  [FP64] Rows: " << f64_matrix->m << ", Cols: " << f64_matrix->n
                << ", NNZ: " << f64_matrix->nnz << "\n";
    }
  } else {
    std::ostringstream ss;
    ss << "ER_graph_V" << vertices << "_D" << static_cast<long>(degree);
    matrix_name = ss.str();
    std::cout << "Generating Erdős-Rényi Graph (V=" << vertices
              << ", Expected Degree=" << degree << ", Seed=" << seed << ")\n";

    if (precision == "float" || precision == "both") {
      f32_matrix =
          std::make_unique<Matrix<float>>(spcraft::GenERGraph<float>(vertices, degree, seed));
      std::cout << "  [FP32] Rows: " << f32_matrix->m << ", Cols: " << f32_matrix->n
                << ", NNZ: " << f32_matrix->nnz << "\n";
    }
    if (precision == "double" || precision == "both") {
      f64_matrix = std::make_unique<Matrix<double>>(
          spcraft::GenERGraph<double>(vertices, degree, seed));
      std::cout << "  [FP64] Rows: " << f64_matrix->m << ", Cols: " << f64_matrix->n
                << ", NNZ: " << f64_matrix->nnz << "\n";
    }
  }
  std::cout << "====================================================\n\n";

  // Register benchmarks with Google Benchmark
  if (f32_matrix) {
    run_spmv_benchmarks<float>(*f32_matrix, matrix_name, thread_counts, min_time, repetitions,
                               seed);
  }
  if (f64_matrix) {
    run_spmv_benchmarks<double>(*f64_matrix, matrix_name, thread_counts, min_time, repetitions,
                                seed);
  }

  // Run all registered Google Benchmarks
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
