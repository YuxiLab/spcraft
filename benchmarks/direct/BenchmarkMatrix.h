#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace spcraft::benchmark
{

struct Matrix {
  int32_t n = 0;
  std::vector<int32_t> col_ptr;
  std::vector<int32_t> row_id;
  std::vector<double> val;
  std::vector<double> rhs;
};

inline Matrix ReadMatrix(const std::string& path)
{
  std::ifstream input(path);
  std::string line;
  std::getline(input, line);
  if (line != "%%MatrixMarket matrix coordinate real general") {
    throw std::runtime_error("Expected a full, real, general Matrix Market matrix");
  }
  do {
    if (!std::getline(input, line)) {
      throw std::runtime_error("Missing matrix dimensions");
    }
  } while (line.empty() || line.front() == '%');
  int64_t m = 0;
  int64_t n = 0;
  int64_t nnz = 0;
  std::istringstream dimensions(line);
  dimensions >> m >> n >> nnz;
  if (!dimensions || m != n || n <= 0 || n >= std::numeric_limits<int32_t>::max() || nnz <= 0 ||
      nnz > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("Invalid dimensions or unsupported 32-bit matrix size");
  }
  std::vector<std::tuple<int32_t, int32_t, double>> entries;
  entries.reserve(static_cast<size_t>(nnz));
  for (int64_t k = 0; k < nnz; ++k) {
    int64_t row = 0;
    int64_t col = 0;
    double value = 0;
    input >> row >> col >> value;
    if (!input || row < 1 || row > n || col < 1 || col > n || !std::isfinite(value)) {
      throw std::runtime_error("Invalid matrix entry");
    }
    entries.emplace_back(static_cast<int32_t>(col - 1), static_cast<int32_t>(row - 1), value);
  }
  std::sort(entries.begin(), entries.end());
  Matrix matrix;
  matrix.n = static_cast<int32_t>(n);
  matrix.col_ptr.assign(static_cast<size_t>(n + 1), 0);
  matrix.rhs.assign(static_cast<size_t>(n), 0);
  for (const auto& [col, row, value] : entries) {
    ++matrix.col_ptr[col + 1];
    matrix.row_id.push_back(row);
    matrix.val.push_back(value);
    matrix.rhs[row] += value;  // Exact solution is the all-ones vector.
  }
  for (int32_t col = 0; col < matrix.n; ++col) {
    matrix.col_ptr[col + 1] += matrix.col_ptr[col];
  }
  return matrix;
}

using Clock = std::chrono::steady_clock;

inline double Seconds(Clock::time_point start)
{
  const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  return elapsed;
}

inline bool Report(const char* solver, const char* ordering, const Matrix& matrix,
                   const std::vector<double>& x, int threads, double analyze, double factor, double solve,
                   int64_t factor_entries, int64_t perturbed_pivots)
{
  std::vector<double> residual = matrix.rhs;
  std::vector<double> row_norm(matrix.n, 0);
  for (int32_t col = 0; col < matrix.n; ++col) {
    for (int32_t k = matrix.col_ptr[col]; k < matrix.col_ptr[col + 1]; ++k) {
      residual[matrix.row_id[k]] -= matrix.val[k] * x[col];
      row_norm[matrix.row_id[k]] += std::abs(matrix.val[k]);
    }
  }
  double r_norm = 0;
  double b_norm = 0;
  double x_norm = 0;
  double a_norm = 0;
  double forward_error = 0;
  for (int32_t row = 0; row < matrix.n; ++row) {
    if (!std::isfinite(x[row]) || !std::isfinite(residual[row])) {
      throw std::runtime_error("Non-finite solution or residual");
    }
    r_norm = std::max(r_norm, std::abs(residual[row]));
    b_norm = std::max(b_norm, std::abs(matrix.rhs[row]));
    x_norm = std::max(x_norm, std::abs(x[row]));
    a_norm = std::max(a_norm, row_norm[row]);
    forward_error = std::max(forward_error, std::abs(x[row] - 1));
  }
  const double backward_error = r_norm / (a_norm * x_norm + b_norm);
  constexpr double backward_tolerance = 1e-10;
  constexpr double forward_tolerance = 1e-7;
  const bool passed = backward_error <= backward_tolerance && forward_error <= forward_tolerance;
  std::cout << std::setprecision(17) << "SPCRAFT_BASELINE {\"solver\":\"" << solver << "\",\"ordering\":\""
            << ordering << "\",\"n\":" << matrix.n << ",\"nnz\":" << matrix.val.size()
            << ",\"threads\":" << threads << ",\"analyze_s\":" << analyze << ",\"factor_s\":" << factor
            << ",\"solve_s\":" << solve << ",\"phase_sum_s\":" << analyze + factor + solve
            << ",\"factor_entries\":" << factor_entries << ",\"perturbed_pivots\":" << perturbed_pivots
            << ",\"backward_error\":" << backward_error << ",\"forward_error\":" << forward_error
            << ",\"relative_residual\":" << r_norm / b_norm << ",\"passed\":" << (passed ? "true" : "false")
            << "}\n";
  return passed;
}

}  // namespace spcraft::benchmark
