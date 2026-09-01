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
#include <string>
#include <utility>
#include <vector>

#include "spcraft.h"

namespace
{

using Index = std::int32_t;
using Offset = std::int64_t;

template <class NT>
using Matrix = spcraft::CsrMatrix<Index, NT, Offset>;

// Generate Erdős-Rényi random graph using Batagelj-Brandes edge-skipping algorithm
template <class NT>
Matrix<NT> make_er_graph(Index vertices, double expected_degree, std::uint64_t seed)
{
  if (vertices < 2 || expected_degree <= 0.0 || expected_degree >= vertices - 1.0) {
    throw std::invalid_argument("expected degree must be in (0, vertices - 1)");
  }

  const double probability = expected_degree / static_cast<double>(vertices - 1);
  const double log_one_minus_p = std::log1p(-probability);
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  std::vector<std::pair<Index, Index>> edges;
  edges.reserve(static_cast<std::size_t>(vertices * expected_degree * 0.525));

  Offset source = 1;
  Offset destination = -1;
  while (source < vertices) {
    const double random_value = uniform(generator);
    destination +=
        1 + static_cast<Offset>(std::floor(std::log1p(-random_value) / log_one_minus_p));
    while (destination >= source && source < vertices) {
      destination -= source;
      ++source;
    }
    if (source < vertices) {
      edges.emplace_back(static_cast<Index>(source), static_cast<Index>(destination));
    }
  }

  std::vector<Offset> degrees(static_cast<std::size_t>(vertices), 0);
  for (const auto& [row, column] : edges) {
    ++degrees[row];
    ++degrees[column];
  }

  Matrix<NT> graph;
  const Offset nonzeros = static_cast<Offset>(edges.size()) * 2;
  graph.Allocate(nonzeros, vertices, vertices);
  graph.row_ptr[0] = 0;
  for (Index row = 0; row < vertices; ++row) {
    graph.row_ptr[row + 1] = graph.row_ptr[row] + degrees[row];
  }

  std::vector<Offset> next(graph.row_ptr, graph.row_ptr + vertices);
  for (const auto& [row, column] : edges) {
    const Offset forward = next[row]++;
    const Offset reverse = next[column]++;
    graph.col_id[forward] = column;
    graph.col_id[reverse] = row;
    graph.val[forward] = static_cast<NT>(1.0);
    graph.val[reverse] = static_cast<NT>(1.0);
  }
  return graph;
}

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
                         const std::vector<int>& thread_counts, double min_time, int repetitions,
                         std::uint64_t seed)
{
  // Allocate input vector x and output vector y
  std::vector<NT> x(static_cast<std::size_t>(A.n));
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (Index i = 0; i < A.n; ++i) {
    x[i] = static_cast<NT>(dist(rng));
  }
  std::vector<NT> y(static_cast<std::size_t>(A.m), 0.0);

  // Warmup run
  omp_set_num_threads(1);
  spcraft::spmv_openmp(A, x.data(), y.data());

  const std::string type_name = std::is_same_v<NT, float> ? "FP32" : "FP64";

  for (int threads : thread_counts) {
    std::string bench_name =
        "SpMV/" + matrix_name + "/" + type_name + "/threads:" + std::to_string(threads);

    auto* benchmark = benchmark::RegisterBenchmark(
        bench_name.c_str(), [&A, &x, &y, threads](benchmark::State& state) {
          omp_set_num_threads(threads);
          for (auto _ : state) {
            spcraft::spmv_openmp(A, x.data(), y.data());
            benchmark::DoNotOptimize(y.data());
            benchmark::ClobberMemory();
          }

          // 2 * nnz floating-point operations per SpMV
          const double total_flops = 2.0 * static_cast<double>(A.nnz) * state.iterations();
          state.SetItemsProcessed(state.iterations() * A.nnz);
          state.counters["GFLOPS"] = benchmark::Counter(total_flops, benchmark::Counter::kIsRate);
          state.counters["NNZ"] = static_cast<double>(A.nnz);
          state.counters["Rows"] = static_cast<double>(A.m);
          state.counters["Cols"] = static_cast<double>(A.n);
          state.counters["Threads"] = static_cast<double>(threads);
        });

    benchmark->MinTime(min_time);
    benchmark->Repetitions(repetitions);
    if (repetitions > 1) {
      benchmark->ReportAggregatesOnly(true);
    }
    benchmark->UseRealTime();
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
    std::cout << "\nGoogle Benchmark options can also be passed (e.g., --benchmark_filter, --benchmark_format=json, --benchmark_out=res.json).\n";
    return 0;
  }

  // Initialize Google Benchmark (this parses & consumes google benchmark specific flags)
  benchmark::Initialize(&argc, argv);

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
    std::cout << "Generating Erdős-Rényi Graph (V=" << vertices << ", Expected Degree=" << degree
              << ", Seed=" << seed << ")\n";

    if (precision == "float" || precision == "both") {
      f32_matrix = std::make_unique<Matrix<float>>(make_er_graph<float>(vertices, degree, seed));
      std::cout << "  [FP32] Rows: " << f32_matrix->m << ", Cols: " << f32_matrix->n
                << ", NNZ: " << f32_matrix->nnz << "\n";
    }
    if (precision == "double" || precision == "both") {
      f64_matrix = std::make_unique<Matrix<double>>(make_er_graph<double>(vertices, degree, seed));
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
