#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "BenchmarkCommon.h"
#include "SpCraft.h"

namespace spcraft::benchmarks
{
template <class IT, class NT>
spcraft::CsrMatrix<IT, NT> FromEigen(const Eigen::SparseMatrix<NT, Eigen::RowMajor>& matrix)
{
  const auto limit = std::numeric_limits<IT>::max();
  if (matrix.rows() > limit || matrix.cols() > limit || matrix.nonZeros() > limit) {
    throw std::overflow_error("input exceeds the selected index type");
  }
  spcraft::CsrMatrix<IT, NT> result;
  result.Allocate(static_cast<IT>(matrix.nonZeros()), static_cast<IT>(matrix.rows()),
                  static_cast<IT>(matrix.cols()));
  std::copy(matrix.outerIndexPtr(), matrix.outerIndexPtr() + matrix.rows() + 1,
            result.row_ptr);
  std::copy(matrix.innerIndexPtr(), matrix.innerIndexPtr() + matrix.nonZeros(), result.col_id);
  std::copy(matrix.valuePtr(), matrix.valuePtr() + matrix.nonZeros(), result.val);
  return result;
}

template <class NT, int Order, class Actual>
double Verify(const Eigen::SparseMatrix<NT, Order>& expected, Actual actual,
              const char* backend)
{
  double error = 0.0;
  double scale = 0.0;
  for (Eigen::Index position = 0; position < expected.nonZeros(); ++position) {
    const double value = actual(position);
    if (!std::isfinite(value)) {
      throw std::runtime_error(std::string(backend) + " produced a nonfinite value");
    }
    error =
        std::max(error, std::abs(value - static_cast<double>(expected.valuePtr()[position])));
    scale = std::max(scale, std::abs(static_cast<double>(expected.valuePtr()[position])));
  }
  if (error > Tolerance<NT>() * std::max(1.0, scale)) {
    throw std::runtime_error(
        fmt::format("{} disagrees with Eigen: max error {}", backend, error));
  }
  return error;
}

template <class IT, class NT>
std::uintmax_t ProductWork(const spcraft::CsrMatrix<IT, NT>& A,
                           const spcraft::CsrMatrix<IT, NT>& B)
{
  std::uintmax_t work = 0;
  const auto limit = static_cast<std::uintmax_t>(std::numeric_limits<IT>::max());
  for (IT row = 0; row < A.m; ++row) {
    for (IT position = A.row_ptr[row]; position < A.row_ptr[row + 1]; ++position) {
      const IT inner = A.col_id[position];
      const auto count = static_cast<std::uintmax_t>(B.row_ptr[inner + 1] - B.row_ptr[inner]);
      // CombBLAS's flop prefix sum uses IT even when SpTuples::nnz is int64_t.
      if (count > limit - work) {
        throw std::overflow_error(
            "product work exceeds CombBLAS index type; use --index int64 --offset int64");
      }
      work += count;
    }
  }
  return work;
}

template <class IT, class NT>
spcraft::CscMatrix<IT, NT> FromEigenColumns(const Eigen::SparseMatrix<NT>& matrix)
{
  const auto limit = std::numeric_limits<IT>::max();
  if (matrix.rows() > limit || matrix.cols() > limit || matrix.nonZeros() > limit) {
    throw std::overflow_error("input exceeds the selected index type");
  }
  spcraft::CscMatrix<IT, NT> result;
  result.Allocate(static_cast<IT>(matrix.nonZeros()), static_cast<IT>(matrix.rows()),
                  static_cast<IT>(matrix.cols()));
  std::copy_n(matrix.outerIndexPtr(), matrix.cols() + 1, result.col_ptr);
  if (matrix.nonZeros()) {
    std::copy_n(matrix.innerIndexPtr(), matrix.nonZeros(), result.row_id);
    std::copy_n(matrix.valuePtr(), matrix.nonZeros(), result.val);
  }
  return result;
}

}  // namespace spcraft::benchmarks
