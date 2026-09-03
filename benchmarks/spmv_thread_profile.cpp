#include <cxxopts.hpp>
#include <omp.h>
#include <sched.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "SpCraft.h"

namespace
{

using Index = std::int32_t;
using Offset = std::int32_t;
using Numeric = double;
using Matrix = spcraft::CsrMatrix<Index, Numeric, Offset>;
using Ring = spcraft::PlusTimesRing<Numeric>;
using Vector = spcraft::DenseVector<Index, Numeric>;

struct ThreadProfile {
  int thread = 0;
  int cpu_start = -1;
  int cpu_end = -1;
  std::int64_t rows = 0;
  std::int64_t nonzeros = 0;
  double average_work_ms = 0.0;
};

std::vector<ThreadProfile> profile_spmv(const Matrix& matrix, const Vector& x, Vector& y,
                                        int requested_threads, int iterations)
{
  omp_set_dynamic(0);
  omp_set_num_threads(requested_threads);

  std::vector<ThreadProfile> profiles(static_cast<std::size_t>(requested_threads));
  int actual_threads = 0;

#pragma omp parallel default(none) shared(matrix, x, y, iterations, profiles, actual_threads)
  {
    const int thread = omp_get_thread_num();
    std::int64_t rows = 0;
    std::int64_t nonzeros = 0;
    double work_seconds = 0.0;
    const int cpu_start = sched_getcpu();

#pragma omp single
    actual_threads = omp_get_num_threads();

    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma omp barrier
      const double start = omp_get_wtime();

#pragma omp for schedule(static) nowait
      for (Index row = 0; row < matrix.m; ++row) {
        Numeric sum{};
        for (Offset position = matrix.row_ptr[row]; position < matrix.row_ptr[row + 1];
             ++position) {
          sum += matrix.val[position] * x.val[matrix.col_id[position]];
        }
        y.val[row] = sum;
        if (iteration == 0) {
          ++rows;
          nonzeros += matrix.row_ptr[row + 1] - matrix.row_ptr[row];
        }
      }

      work_seconds += omp_get_wtime() - start;
    }

    profiles[thread] = ThreadProfile{thread, cpu_start, sched_getcpu(),
                                     rows,   nonzeros,  work_seconds * 1000.0 / iterations};
  }

  profiles.resize(static_cast<std::size_t>(actual_threads));
  return profiles;
}

}  // namespace

int main(int argc, char** argv)
{
  cxxopts::Options options(argv[0],
                           "Report per-thread work and load balance for SPCraft SpMV");
  options.add_options()("m,matrix", "Matrix Market file", cxxopts::value<std::string>())(
      "t,threads", "OpenMP thread count", cxxopts::value<int>()->default_value("16"))(
      "i,iterations", "Profiled iterations", cxxopts::value<int>()->default_value("20"))(
      "h,help", "Print help");

  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0 || arguments.count("matrix") == 0) {
    std::cout << options.help() << '\n';
    return arguments.count("help") != 0 ? 0 : 2;
  }

  const auto path = std::filesystem::path(arguments["matrix"].as<std::string>());
  const int threads = arguments["threads"].as<int>();
  const int iterations = arguments["iterations"].as<int>();
  if (threads <= 0 || iterations <= 0) {
    std::cerr << "error: threads and iterations must be positive\n";
    return 2;
  }

  try {
    auto coo = spcraft::CooMatrix<Index, Numeric, Offset>::FromMatrixMarket(path.string());
    Matrix matrix = coo.ToCsr();
    Vector x(matrix.n);
    Vector y(matrix.m);
    std::fill(x.begin(), x.end(), 1.0);

    spcraft::spmv_openmp<Ring>(matrix, x, y);
    const auto profiles = profile_spmv(matrix, x, y, threads, iterations);

    const double mean_time = std::accumulate(profiles.begin(), profiles.end(), 0.0,
                                             [](double sum, const ThreadProfile& profile) {
                                               return sum + profile.average_work_ms;
                                             }) /
                             profiles.size();
    const double max_time =
        std::max_element(profiles.begin(), profiles.end(),
                         [](const ThreadProfile& left, const ThreadProfile& right) {
                           return left.average_work_ms < right.average_work_ms;
                         })
            ->average_work_ms;
    const double mean_nnz = static_cast<double>(matrix.nnz) / profiles.size();
    const auto max_nnz =
        std::max_element(profiles.begin(), profiles.end(),
                         [](const ThreadProfile& left, const ThreadProfile& right) {
                           return left.nonzeros < right.nonzeros;
                         })
            ->nonzeros;

    std::cout << "matrix,thread,cpu_start,cpu_end,rows,nnz,avg_work_ms,estimated_wait_ms\n";
    std::cout << std::setprecision(10);
    for (const auto& profile : profiles) {
      std::cout << path.stem().string() << ',' << profile.thread << ',' << profile.cpu_start
                << ',' << profile.cpu_end << ',' << profile.rows << ',' << profile.nonzeros
                << ',' << profile.average_work_ms << ',' << max_time - profile.average_work_ms
                << '\n';
    }

    std::cerr << std::setprecision(5) << "matrix=" << path.stem().string()
              << " rows=" << matrix.m << " nnz=" << matrix.nnz
              << " threads=" << profiles.size() << " mean_work_ms=" << mean_time
              << " max_work_ms=" << max_time << " time_imbalance=" << max_time / mean_time
              << " nnz_imbalance=" << static_cast<double>(max_nnz) / mean_nnz << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
