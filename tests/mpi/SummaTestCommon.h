#pragma once

#include <tuple>
#include <type_traits>

#include <Eigen/Dense>
#include <gtest/gtest.h>

// Driver helpers remain at global scope, matching the MPI test entry points.
using namespace spcraft;

using GridPtr = std::shared_ptr<const CommGrid2D>;

template <class NT>
using Ref = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>;

using DistributedSummaTypes =
    testing::Types<std::tuple<int32_t, float, int32_t, DistSpCSC<int32_t, float, int32_t>>,
                   std::tuple<int32_t, float, int32_t, DistSpDCSC<int32_t, float, int32_t>>,
                   std::tuple<int32_t, double, int32_t, DistSpCSC<int32_t, double, int32_t>>,
                   std::tuple<int32_t, double, int32_t, DistSpDCSC<int32_t, double, int32_t>>,
                   std::tuple<int32_t, float, int64_t, DistSpCSC<int32_t, float, int64_t>>,
                   std::tuple<int32_t, float, int64_t, DistSpDCSC<int32_t, float, int64_t>>,
                   std::tuple<int32_t, double, int64_t, DistSpCSC<int32_t, double, int64_t>>,
                   std::tuple<int32_t, double, int64_t, DistSpDCSC<int32_t, double, int64_t>>,
                   std::tuple<int64_t, float, int64_t, DistSpCSC<int64_t, float, int64_t>>,
                   std::tuple<int64_t, float, int64_t, DistSpDCSC<int64_t, float, int64_t>>,
                   std::tuple<int64_t, double, int64_t, DistSpCSC<int64_t, double, int64_t>>,
                   std::tuple<int64_t, double, int64_t, DistSpDCSC<int64_t, double, int64_t>>>;

template <class IT, class NT, class OT, class Matrix, class Predicate>
Matrix MakeSummaInput(const Ref<NT>& a, GridPtr grid, IT m, IT n, Predicate owns)
{
  Matrix result(grid, m, n);
  const IT row_extent = m / grid->Rows() + (m % grid->Rows() != 0);
  const IT col_extent = n / grid->Cols() + (n % grid->Cols() != 0);

  // Enumerate global entries independently of the library's local/global formulas.
  OT count = 0;
  for (IT row = 0; row < m; ++row) {
    for (IT col = 0; col < n; ++col) {
      if (a(row, col) != NT{0} && result.Owns(row, col) && owns(row, col)) ++count;
    }
  }
  CooMatrix<IT, NT, OT> local;
  local.Allocate(count, result.LocalRows(), result.LocalCols());
  OT position = 0;
  for (IT row = 0; row < m; ++row) {
    for (IT col = 0; col < n; ++col) {
      if (a(row, col) == NT{0}) continue;
      if (!result.Owns(row, col) || !owns(row, col)) continue;
      const IT local_row = row - grid->Row() * row_extent;
      const IT local_col = col - grid->Col() * col_extent;
      local.entries[position++] = {local_row, local_col, a(row, col)};
    }
  }

  // Each format uses its public native conversion, retaining move-only ownership.
  CscMatrix<IT, NT, OT> csc = local.ToCsc();
  if constexpr (std::is_same_v<typename Matrix::MatrixType, CscMatrix<IT, NT, OT>>) {
    result.local = std::move(csc);
  } else {
    result.local = DcscMatrix<IT, NT, OT>::FromCsc(csc);
  }
  return result;
}

template <class IT, class NT, class OT>
void CheckSummaValues(const DistSpCOO<IT, NT, OT>& actual, const Ref<NT>& expected)
{
  ASSERT_EQ(actual.m, expected.rows());
  ASSERT_EQ(actual.n, expected.cols());
  const CommGrid2D& grid = *actual.commgrid;
  Ref<NT> values = Ref<NT>::Zero(actual.local.m, actual.local.n);
  // Check sorted uniqueness as well as values, including retained structural zeros.
  for (OT p = 0; p < actual.local.nnz; ++p) {
    const TupleEntry<IT, NT>& entry = actual.local.entries[p];
    ASSERT_GE(entry.row, 0);
    ASSERT_LT(entry.row, actual.local.m);
    ASSERT_GE(entry.col, 0);
    ASSERT_LT(entry.col, actual.local.n);
    if (p > 0) {
      const TupleEntry<IT, NT>& previous = actual.local.entries[p - 1];
      EXPECT_TRUE(previous.col < entry.col || (previous.col == entry.col && previous.row < entry.row));
    }
    values(entry.row, entry.col) = entry.val;
  }
  constexpr NT kTolerance = std::is_same_v<NT, float> ? NT{1e-5} : NT{1e-12};
  for (IT local_row = 0; local_row < actual.local.m; ++local_row) {
    const IT row = actual.GlobalRow(local_row);
    for (IT local_col = 0; local_col < actual.local.n; ++local_col) {
      const IT col = actual.GlobalCol(local_col);
      EXPECT_NEAR(values(local_row, local_col), expected(row, col), kTolerance)
          << "rank=" << grid.Rank() << " global=(" << row << ',' << col << ')';
    }
  }
}
