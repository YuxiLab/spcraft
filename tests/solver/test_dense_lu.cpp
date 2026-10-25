#include "LuFixture.h"

#include <limits>

namespace spcraft::solver::test
{
namespace
{

using DenseTestMatrix = DenseMatrix<Index, Value>;

DenseTestMatrix MakeDenseExample()
{
  // 1. Store the worked example by columns: A=[[4,1,1],[2,5,0],[0,1,6]].
  DenseTestMatrix A(3, 3);
  const Value values[] = {4, 2, 0, 1, 5, 1, 1, 0, 6};
  std::copy(std::begin(values), std::end(values), A.val);
  return A;
}

dense::Factors<Index, Value> MakeDenseFactors()
{
  // 1. Supply known packed factors so solve checkpoints do not depend on Factorize.
  dense::Factors<Index, Value> factors;
  factors.lu.Allocate(3, 3);
  const Value packed[] = {4, 0.5, 0, 1, 4.5, 2.0 / 9.0, 1, -0.5, 55.0 / 9.0};
  std::copy(std::begin(packed), std::end(packed), factors.lu.val);
  factors.status.code = LuCode::Success;
  return factors;
}

}  // namespace

TEST(W01DenseForwardSolve, PackedUnitLowerAndPreservedInputs)
{
  // 1. Predict y from L*y=b without using a factorization implementation.
  const auto factors = MakeDenseFactors();
  const std::vector<Value> original(factors.lu.val, factors.lu.val + 9);
  Vector rhs(3);
  rhs[0] = 9;
  rhs[1] = 12;
  rhs[2] = 20;
  Vector y(3);
  dense::ForwardSolve(factors.lu, rhs, y);
  EXPECT_NEAR(y[0], 9, kTolerance);
  EXPECT_NEAR(y[1], 7.5, kTolerance);
  EXPECT_NEAR(y[2], 55.0 / 3.0, kTolerance);
  EXPECT_EQ(std::vector<Value>(rhs.val, rhs.val + 3), (std::vector<Value>{9, 12, 20}));
  EXPECT_EQ(std::vector<Value>(factors.lu.val, factors.lu.val + 9), original);
}

TEST(W01DenseForwardSolve, Empty)
{
  // 1. An empty row loop leaves empty output unchanged.
  DenseTestMatrix empty;
  Vector empty_rhs;
  Vector output;
  EXPECT_NO_THROW(dense::ForwardSolve(empty, empty_rhs, output));
}

TEST(W02DenseBackwardSolve, PackedUpperAndPreservedInputs)
{
  // 1. Supply the forward result directly so BackwardSolve is independent of ForwardSolve.
  const auto factors = MakeDenseFactors();
  const std::vector<Value> original(factors.lu.val, factors.lu.val + 9);
  Vector rhs(3);
  rhs[0] = 9;
  rhs[1] = 7.5;
  rhs[2] = 55.0 / 3.0;
  Vector x(3);
  dense::BackwardSolve(factors.lu, rhs, x);
  for (Index i = 0; i < 3; ++i) EXPECT_NEAR(x[i], i + 1, kTolerance);
  EXPECT_EQ(std::vector<Value>(rhs.val, rhs.val + 3), (std::vector<Value>{9, 7.5, 55.0 / 3.0}));
  EXPECT_EQ(std::vector<Value>(factors.lu.val, factors.lu.val + 9), original);
}

TEST(W02DenseBackwardSolve, EmptyAndRejectedDiagonals)
{
  // 1. Exercise n=0 and zero/nonfinite U diagonals before division.
  DenseTestMatrix empty;
  Vector empty_rhs;
  Vector x;
  EXPECT_NO_THROW(dense::BackwardSolve(empty, empty_rhs, x));
  DenseTestMatrix lu(1, 1, Value{0});
  Vector rhs(1, 1);
  x.Allocate(1);
  EXPECT_THROW(dense::BackwardSolve(lu, rhs, x), std::runtime_error);
  lu(0, 0) = std::numeric_limits<Value>::infinity();
  EXPECT_THROW(dense::BackwardSolve(lu, rhs, x), std::runtime_error);
}

TEST(DenseRightLookingFactorize, PackedFactorsAndPreservedInput)
{
  // 1. Compare every packed coefficient with the independent hand calculation.
  const auto A = MakeDenseExample();
  const std::vector<Value> original(A.val, A.val + 9);
  const auto expected = MakeDenseFactors();
  auto factors = dense::RightLookingFactorize(A);
  ASSERT_EQ(factors.status.code, LuCode::Success);
  EXPECT_EQ(factors.status.pivot_column, -1);
  EXPECT_EQ(factors.lu.m, 3);
  EXPECT_EQ(factors.lu.n, 3);
  ASSERT_EQ(factors.lu.size(), 9);
  EXPECT_TRUE(factors.lu.memowned);
  EXPECT_NE(factors.lu.val, A.val);
  for (Index i = 0; i < 9; ++i) EXPECT_NEAR(factors.lu.val[i], expected.lu.val[i], kTolerance);
  // 2. Reconstruct L*U from the packed triangles in the original coordinates.
  for (Index col = 0; col < 3; ++col) {
    for (Index row = 0; row < 3; ++row) {
      Value sum = 0;
      for (Index k = 0; k < 3; ++k) {
        const Value lower = row == k ? 1 : (row > k ? factors.lu(row, k) : 0);
        const Value upper = k <= col ? factors.lu(k, col) : 0;
        sum += lower * upper;
      }
      EXPECT_NEAR(sum, A(row, col), kTolerance);
    }
  }
  EXPECT_EQ(std::vector<Value>(A.val, A.val + 9), original);
  factors.lu(0, 0) = 99;
  EXPECT_EQ(A(0, 0), 4);
}

TEST(DenseRightLookingFactorize, Empty)
{
  // 1. No pivot iterations are needed for a successful empty factorization.
  DenseTestMatrix empty;
  const auto factors = dense::RightLookingFactorize(empty);
  EXPECT_EQ(factors.status.code, LuCode::Success);
  EXPECT_EQ(factors.status.pivot_column, -1);
  EXPECT_EQ(factors.lu.size(), 0);
}

TEST(DenseRightLookingFactorize, FirstLaterAndToleranceBoundaryPivots)
{
  // 1. A nonsingular matrix can still have a rejected first no-pivot diagonal.
  DenseTestMatrix A(2, 2);
  const Value first[] = {0, 1, 1, 0};
  std::copy(std::begin(first), std::end(first), A.val);
  auto factors = dense::RightLookingFactorize(A);
  EXPECT_EQ(factors.status.code, LuCode::ZeroPivot);
  EXPECT_EQ(factors.status.pivot_column, 0);
  // 2. Check the updated later pivot, then the inclusive tolerance comparison.
  const Value later[] = {2, 1, 4, 2};
  std::copy(std::begin(later), std::end(later), A.val);
  factors = dense::RightLookingFactorize(A);
  EXPECT_EQ(factors.status.code, LuCode::ZeroPivot);
  EXPECT_EQ(factors.status.pivot_column, 1);
  DenseTestMatrix diagonal(1, 1, Value{2});
  factors = dense::RightLookingFactorize(diagonal, LuOptions<Value>{2});
  EXPECT_EQ(factors.status.code, LuCode::ZeroPivot);
  EXPECT_EQ(factors.status.pivot_column, 0);
}

TEST(DenseRightLookingFactorize, PaddedInputClonesOnlyLogicalEntries)
{
  // 1. Padding contains NaNs that are outside the mathematical 3-by-3 input.
  const auto A = MakeDenseExample();
  Value buffer[13];
  std::fill(std::begin(buffer), std::end(buffer), std::numeric_limits<Value>::quiet_NaN());
  DenseTestMatrix view(buffer, 3, 3, 5);
  for (Index col = 0; col < 3; ++col) {
    for (Index row = 0; row < 3; ++row) view(row, col) = A(row, col);
  }
  const auto factors = dense::RightLookingFactorize(view);
  const auto expected = MakeDenseFactors();
  // 2. The result is an independent packed owner, not a copy of padding slots.
  ASSERT_EQ(factors.status.code, LuCode::Success);
  ASSERT_EQ(factors.lu.m, 3);
  ASSERT_EQ(factors.lu.n, 3);
  EXPECT_EQ(factors.lu.ld, 3);
  EXPECT_TRUE(factors.lu.memowned);
  EXPECT_NE(factors.lu.val, buffer);
  for (Index col = 0; col < 3; ++col) {
    for (Index row = 0; row < 3; ++row) {
      EXPECT_NEAR(factors.lu(row, col), expected.lu(row, col), kTolerance);
      EXPECT_EQ(view(row, col), A(row, col));
    }
  }
  EXPECT_TRUE(std::isnan(buffer[3]));
  EXPECT_TRUE(std::isnan(buffer[8]));
}

TEST(W04DenseSolve, KnownFactorsReusableRhsAndOriginalResidual)
{
  // 1. Solve with supplied factors so Solve does not depend on RightLookingFactorize.
  const auto factors = MakeDenseFactors();
  const std::vector<Value> original(factors.lu.val, factors.lu.val + 9);
  const auto A = MakeDenseExample();
  Vector rhs(3);
  Vector x(3);
  for (Value scale : {1, 2}) {
    rhs[0] = 9 * scale;
    rhs[1] = 12 * scale;
    rhs[2] = 20 * scale;
    const std::vector<Value> saved(rhs.val, rhs.val + 3);
    dense::Solve(factors, rhs, x);
    for (Index row = 0; row < 3; ++row) {
      EXPECT_NEAR(x[row], scale * (row + 1), kTolerance);
      Value product = 0;
      for (Index col = 0; col < 3; ++col) product += A(row, col) * x[col];
      EXPECT_NEAR(product, rhs[row], kTolerance);
    }
    EXPECT_EQ(std::vector<Value>(rhs.val, rhs.val + 3), saved);
  }
  EXPECT_EQ(std::vector<Value>(factors.lu.val, factors.lu.val + 9), original);
  EXPECT_EQ(factors.status.code, LuCode::Success);
}

TEST(W04DenseSolve, Empty)
{
  // 1. Successful empty factors require no substitution work.
  dense::Factors<Index, Value> factors;
  Vector empty;
  Vector x;
  factors.status.code = LuCode::Success;
  EXPECT_NO_THROW(dense::Solve(factors, empty, x));
}

TEST(DenseLu, RightLookingFactorizeThenSolve)
{
  // 1. Integrate the dense driver only after all four isolated checkpoints work.
  const auto A = MakeDenseExample();
  const auto factors = dense::RightLookingFactorize(A);
  ASSERT_EQ(factors.status.code, LuCode::Success);
  Vector rhs(3);
  rhs[0] = 9;
  rhs[1] = 12;
  rhs[2] = 20;
  Vector x(3);
  dense::Solve(factors, rhs, x);
  for (Index i = 0; i < 3; ++i) EXPECT_NEAR(x[i], i + 1, kTolerance);
}

}  // namespace spcraft::solver::test
