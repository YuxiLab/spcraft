#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include <omp.h>

struct CSR {
  std::int64_t rows{};
  std::int64_t cols{};
  std::vector<std::int64_t> row_ptr;
  std::vector<std::int32_t> col_idx;
  std::vector<double> values;
};

void spmv_serial(const CSR& a, const double* x, double* y) {
  for (std::int64_t i = 0; i < a.rows; ++i) {
    double sum = 0.0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      sum += a.values[k] * x[a.col_idx[k]];
    }
    y[i] = sum;
  }
}

void spmv_openmp(const CSR& a, const double* x, double* y) {
  // schedule(runtime) permits controlled experiments with
  // OMP_SCHEDULE=static, dynamic, guided, and chosen chunk sizes.
#pragma omp parallel for schedule(runtime) default(none) shared(a, x, y)
  for (std::int64_t i = 0; i < a.rows; ++i) {
    double sum = 0.0;  // private because it is declared inside the loop body
#pragma omp simd reduction(+ : sum)
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      sum += a.values[k] * x[a.col_idx[k]];
    }
    y[i] = sum;  // each iteration owns a distinct output element
  }
}

CSR five_point_laplacian(std::int32_t nx, std::int32_t ny) {
  CSR a;
  a.rows = static_cast<std::int64_t>(nx) * ny;
  a.cols = a.rows;
  a.row_ptr.reserve(a.rows + 1);
  a.row_ptr.push_back(0);

  auto append = [&](std::int32_t column, double value) {
    a.col_idx.push_back(column);
    a.values.push_back(value);
  };

  for (std::int32_t j = 0; j < ny; ++j) {
    for (std::int32_t i = 0; i < nx; ++i) {
      const std::int32_t p = j * nx + i;
      if (j > 0) append(p - nx, -1.0);
      if (i > 0) append(p - 1, -1.0);
      append(p, 4.0);
      if (i + 1 < nx) append(p + 1, -1.0);
      if (j + 1 < ny) append(p + nx, -1.0);
      a.row_ptr.push_back(static_cast<std::int64_t>(a.values.size()));
    }
  }
  return a;
}

int main() {
  const CSR a = five_point_laplacian(512, 512);
  std::vector<double> x(a.cols);
  std::iota(x.begin(), x.end(), 1.0);
  std::vector<double> reference(a.rows);
  std::vector<double> parallel(a.rows);

  spmv_serial(a, x.data(), reference.data());
  spmv_openmp(a, x.data(), parallel.data());

  double max_error = 0.0;
  for (std::int64_t i = 0; i < a.rows; ++i) {
    max_error = std::max(max_error, std::abs(reference[i] - parallel[i]));
  }
  std::cout << "rows=" << a.rows << " nnz=" << a.values.size()
            << " threads=" << omp_get_max_threads()
            << " max_error=" << max_error << '\n';
  assert(max_error == 0.0);
}
