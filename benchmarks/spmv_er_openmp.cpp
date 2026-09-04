#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

#include "SpCraft.h"

namespace
{

using Index = std::int32_t;
using Offset = std::int64_t;
using Matrix = spcraft::CsrMatrix<Index, double, Offset>;
using Ring = spcraft::PlusTimesRing<double>;
using Vector = spcraft::DenseVector<Index, double>;

void spmv_serial(const Matrix& graph, const Vector& x, Vector& y)
{
  for (Index row = 0; row < graph.m; ++row) {
    double sum = 0.0;
    for (Offset position = graph.row_ptr[row]; position < graph.row_ptr[row + 1]; ++position) {
      sum += graph.val[position] * x.val[graph.col_id[position]];
    }
    y.val[row] = sum;
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

    Matrix graph = spcraft::GenERGraph<double, Index, Offset>(vertices, expected_degree, seed);
    Vector x(vertices);
    for (Index i = 0; i < vertices; ++i) {
      x.val[i] = 0.5 + static_cast<double>((i * 17) % 101) / 101.0;
    }
    Vector reference(vertices);
    Vector result(vertices);
    spmv_serial(graph, x, reference);

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
        spcraft::spmv_openmp<Ring>(graph, x, result);
      }

      std::vector<double> timings;
      timings.reserve(static_cast<std::size_t>(samples));
      for (int sample = 0; sample < samples; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
          spcraft::spmv_openmp<Ring>(graph, x, result);
        }
        const auto stop = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;
        timings.push_back(elapsed.count() / iterations);
      }

      double max_error = 0.0;
      for (Index row = 0; row < vertices; ++row) {
        max_error = std::max(max_error, std::abs(reference.val[row] - result.val[row]));
      }
      if (max_error != 0.0) {
        throw std::runtime_error("parallel result differs from the serial reference");
      }

      const double milliseconds = median(std::move(timings));
      const double gflops = 2.0 * static_cast<double>(graph.nnz) / (milliseconds * 1.0e6);
      const double checksum = std::accumulate(result.val, result.val + result.n, 0.0);
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
