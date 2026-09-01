#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

#include "mtspmv.h"

namespace
{

using Index = std::int32_t;
using Offset = std::int64_t;
using Matrix = spcraft::CsrMatrix<Index, double, Offset>;

Matrix make_er_graph(Index vertices, double expected_degree, std::uint64_t seed)
{
  if (vertices < 2 || expected_degree <= 0.0 || expected_degree >= vertices - 1.0) {
    throw std::invalid_argument("expected degree must be in (0, vertices - 1)");
  }

  // Batagelj-Brandes edge skipping samples the undirected G(n, p) model
  // without examining all O(n^2) possible edges.
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
    destination += 1 + static_cast<Offset>(
                           std::floor(std::log1p(-random_value) / log_one_minus_p));
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

  Matrix graph;
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
    graph.val[forward] = 1.0;
    graph.val[reverse] = 1.0;
  }
  return graph;
}

void spmv_serial(const Matrix& graph, const double* x, double* y)
{
  for (Index row = 0; row < graph.m; ++row) {
    double sum = 0.0;
    for (Offset position = graph.row_ptr[row]; position < graph.row_ptr[row + 1]; ++position) {
      sum += graph.val[position] * x[graph.col_id[position]];
    }
    y[row] = sum;
  }
}

double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[middle - 1] + values[middle]) * 0.5;
  }
  return values[middle];
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 7) {
    std::cerr << "usage: spmv_er_openmp <vertices> <expected-degree> <iterations> "
                 "<samples> <seed> <threads> [threads ...]\n";
    return 2;
  }

  try {
    const Index vertices = static_cast<Index>(std::stoll(argv[1]));
    const double expected_degree = std::stod(argv[2]);
    const int iterations = std::stoi(argv[3]);
    const int samples = std::stoi(argv[4]);
    const std::uint64_t seed = std::stoull(argv[5]);
    if (iterations <= 0 || samples <= 0) {
      throw std::invalid_argument("iterations and samples must be positive");
    }

    Matrix graph = make_er_graph(vertices, expected_degree, seed);
    std::vector<double> x(static_cast<std::size_t>(vertices));
    for (Index i = 0; i < vertices; ++i) {
      x[i] = 0.5 + static_cast<double>((i * 17) % 101) / 101.0;
    }
    std::vector<double> reference(static_cast<std::size_t>(vertices));
    std::vector<double> result(static_cast<std::size_t>(vertices));
    spmv_serial(graph, x.data(), reference.data());

    omp_set_dynamic(0);
    std::cout << "vertices,nnz,realized_degree,threads,median_ms,gflops,max_error,checksum\n";
    std::cout << std::setprecision(10);
    for (int argument = 6; argument < argc; ++argument) {
      const int threads = std::stoi(argv[argument]);
      if (threads <= 0) {
        throw std::invalid_argument("thread counts must be positive");
      }
      omp_set_num_threads(threads);

      for (int warmup = 0; warmup < 3; ++warmup) {
        spcraft::spmv_openmp(graph, x.data(), result.data());
      }

      std::vector<double> timings;
      timings.reserve(static_cast<std::size_t>(samples));
      for (int sample = 0; sample < samples; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
          spcraft::spmv_openmp(graph, x.data(), result.data());
        }
        const auto stop = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;
        timings.push_back(elapsed.count() / iterations);
      }

      double max_error = 0.0;
      for (Index row = 0; row < vertices; ++row) {
        max_error = std::max(max_error, std::abs(reference[row] - result[row]));
      }
      if (max_error != 0.0) {
        throw std::runtime_error("parallel result differs from the serial reference");
      }

      const double milliseconds = median(std::move(timings));
      const double gflops = 2.0 * static_cast<double>(graph.nnz) / (milliseconds * 1.0e6);
      const double checksum = std::accumulate(result.begin(), result.end(), 0.0);
      const double realized_degree = static_cast<double>(graph.nnz) / vertices;
      std::cout << vertices << ',' << graph.nnz << ',' << realized_degree << ',' << threads
                << ',' << milliseconds << ',' << gflops << ',' << max_error << ',' << checksum
                << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
