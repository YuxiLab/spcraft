#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>

#include "SpCraft.h"

namespace spcraft::solver::test
{

using Index = int32_t;
using Value = double;
using Matrix = CscMatrix<Index, Value>;
using Vector = DenseVector<Index, Value>;
constexpr Value kTolerance = 1e-12;

//! A four-cycle with diagonal (6,7,8,9); elimination creates fill at (2,1).
inline Matrix MakeMatrix()
{
  Matrix A;
  A.Allocate(12, 4, 4);
  const Index ptr[] = {0, 3, 6, 9, 12};
  const Index rows[] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  const Value values[] = {6, 1, 1, 1, 7, 1, 1, 8, 1, 1, 1, 9};
  std::copy(std::begin(ptr), std::end(ptr), A.col_ptr);
  std::copy(std::begin(rows), std::end(rows), A.row_id);
  std::copy(std::begin(values), std::end(values), A.val);
  return A;
}

inline LuSymbolic<Index> MakeScalarPlan(bool permuted = true)
{
  LuSymbolic<Index> plan;
  plan.n = 4;
  if (permuted) plan.order = {2, 0, 3, 1};
  plan.lower_col_ptr = {0, 3, 6, 8, 9};
  plan.lower_row_id = {0, 1, 2, 1, 2, 3, 2, 3, 3};
  plan.upper_col_ptr = {0, 1, 3, 6, 9};
  plan.upper_row_id = {0, 0, 1, 0, 1, 2, 1, 2, 3};
  return plan;
}

inline SupernodalSymbolic<Index> MakeBlockPlan()
{
  SupernodalSymbolic<Index> plan;
  plan.scalar = MakeScalarPlan();
  plan.panels = {{0, 1, {1, 2}, {1, 2}, {}}, {1, 2, {2, 3}, {2, 3}, {0}}, {2, 4, {}, {}, {0, 1}}};
  return plan;
}

inline LuSymbolic<Index> MakeZeroPivotPlan()
{
  LuSymbolic<Index> plan;
  plan.n = 2;
  plan.lower_col_ptr = {0, 2, 3};
  plan.lower_row_id = {0, 1, 1};
  plan.upper_col_ptr = {0, 1, 3};
  plan.upper_row_id = {0, 0, 1};
  return plan;
}

//! Nonsingular [[0,1],[1,0]] requires pivoting and must be rejected by this solver.
inline Matrix MakeZeroPivotMatrix()
{
  Matrix A;
  A.Allocate(4, 2, 2);
  const Index ptr[] = {0, 2, 4};
  const Index rows[] = {0, 1, 0, 1};
  const Value values[] = {0, 1, 1, 0};
  std::copy(std::begin(ptr), std::end(ptr), A.col_ptr);
  std::copy(std::begin(rows), std::end(rows), A.row_id);
  std::copy(std::begin(values), std::end(values), A.val);
  return A;
}

inline Vector MakeRhs()
{
  Vector b(4);
  const Value values[] = {11, 19, 29, 41};  // A * (1,2,3,4)^T in original coordinates.
  std::copy(std::begin(values), std::end(values), b.val);
  return b;
}

inline void CheckSolution(const Vector& x, Value scale = 1)
{
  ASSERT_EQ(x.n, 4);
  for (Index i = 0; i < x.n; ++i) {
    ASSERT_TRUE(std::isfinite(x.val[i]));
    EXPECT_NEAR(x.val[i], scale * (i + 1), kTolerance);
  }
}

inline std::vector<Value> Dense(const Matrix& A)
{
  std::vector<Value> values(static_cast<size_t>(A.n) * A.n, 0);
  for (Index col = 0; col < A.n; ++col) {
    for (Index k = A.col_ptr[col]; k < A.col_ptr[col + 1]; ++k) {
      values[A.row_id[k] + col * A.n] = A.val[k];
    }
  }
  return values;
}

//! Independent small dense reconstruction; this is a checker, not an LU algorithm.
inline void CheckProduct(const Matrix& A, const std::vector<Index>& order, const std::vector<Value>& lower,
                         const std::vector<Value>& upper)
{
  const auto dense = Dense(A);
  ASSERT_EQ(lower.size(), dense.size());
  ASSERT_EQ(upper.size(), dense.size());
  for (Index col = 0; col < A.n; ++col) {
    for (Index row = 0; row < A.n; ++row) {
      Value sum = 0;
      for (Index k = 0; k < A.n; ++k) sum += lower[row + k * A.n] * upper[k + col * A.n];
      const Index old_row = order.empty() ? row : order[row];
      const Index old_col = order.empty() ? col : order[col];
      EXPECT_NEAR(sum, dense[old_row + old_col * A.n], kTolerance);
    }
    EXPECT_NEAR(lower[col + col * A.n], 1, kTolerance);
  }
}

inline void CheckBlocks(const Matrix& A, const supernodal::Factors<Index, Value>& factors)
{
  ASSERT_EQ(factors.n, A.n);
  std::vector<Value> lower(static_cast<size_t>(A.n) * A.n, 0);
  std::vector<Value> upper(lower.size(), 0);
  for (const auto& panel : factors.panels) {
    const Index width = panel.last - panel.first;
    ASSERT_GT(width, 0);
    ASSERT_EQ(panel.diagonal.n, width * width);
    ASSERT_EQ(panel.lower.n, static_cast<Index>(panel.rows.size()) * width);
    ASSERT_EQ(panel.upper.n, width * static_cast<Index>(panel.columns.size()));
    for (Index col = 0; col < width; ++col) {
      for (Index row = 0; row < width; ++row) {
        const auto index = panel.first + row + (panel.first + col) * A.n;
        if (row == col) lower[index] = 1;
        if (row > col) lower[index] = panel.diagonal.val[row + col * width];
        if (row <= col) upper[index] = panel.diagonal.val[row + col * width];
      }
      for (size_t row = 0; row < panel.rows.size(); ++row) {
        lower[panel.rows[row] + (panel.first + col) * A.n] = panel.lower.val[row + col * panel.rows.size()];
      }
    }
    for (size_t col = 0; col < panel.columns.size(); ++col) {
      for (Index row = 0; row < width; ++row) {
        upper[panel.first + row + panel.columns[col] * A.n] = panel.upper.val[row + col * width];
      }
    }
  }
  CheckProduct(A, factors.order, lower, upper);
}

}  // namespace spcraft::solver::test
