#include "LuFixture.h"

#include <limits>

namespace spcraft::solver::test
{
namespace
{

using DenseTestMatrix = DenseMatrix<Index, Value>;

dense::Factors<Index, Value> MakeDenseFactors()
{
  // Use the numerical fixture from docs/solver/visuals/left-looking-lu-8x8.html.
  constexpr Index n = 8;
  dense::Factors<Index, Value> factors;
  factors.lu.Allocate(n, n);
  for (Index col = 0; col < n; ++col) {
    for (Index row = 0; row < n; ++row) {
      if (row > col) {
        factors.lu(row, col) = 0.25 + 0.25 * ((row + 2 * col) % 3);
      } else if (row == col) {
        factors.lu(row, col) = 2.4 + 0.04 * row;
      } else {
        factors.lu(row, col) = 0.24 + 0.04 * ((3 * row + 5 * col) % 20);
      }
    }
  }
  factors.status.code = LuCode::Success;
  return factors;
}

Value PackedProductEntry(const DenseTestMatrix& packed, Index row, Index col)
{
  Value sum = 0;
  for (Index k = 0; k <= std::min(row, col); ++k) {
    const Value lower = row == k ? 1 : packed(row, k);
    sum += lower * packed(k, col);
  }
  return sum;
}

DenseTestMatrix MakeDenseExample(const DenseTestMatrix& packed)
{
  DenseTestMatrix A(packed.m, packed.n);
  for (Index col = 0; col < A.n; ++col) {
    for (Index row = 0; row < A.m; ++row) A(row, col) = PackedProductEntry(packed, row, col);
  }
  return A;
}

Vector MakeDenseRhs(const DenseTestMatrix& A, Value scale)
{
  Vector rhs(A.m);
  for (Index row = 0; row < A.m; ++row) {
    rhs[row] = 0;
    for (Index col = 0; col < A.n; ++col) rhs[row] += A(row, col) * (scale * (col + 1));
  }
  return rhs;
}

}  // namespace

TEST(ForwardSolve, NormalInput)
{
  const auto factors = MakeDenseFactors();
  const auto& lu = factors.lu;
  const std::vector<Value> original(lu.val, lu.val + lu.size());
  Vector rhs(lu.n);
  // Form L*y for y = [1, 2, ..., 8], using the implicit unit diagonal of L.
  for (Index row = 0; row < lu.n; ++row) {
    rhs[row] = row + 1;
    for (Index col = 0; col < row; ++col) rhs[row] += lu(row, col) * (col + 1);
  }
  const std::vector<Value> saved(rhs.val, rhs.val + rhs.n);
  Vector y(lu.n);
  dense::ForwardSolve(lu, rhs, y);
  for (Index row = 0; row < lu.n; ++row) EXPECT_NEAR(y[row], row + 1, kTolerance);
  EXPECT_EQ(std::vector<Value>(rhs.val, rhs.val + rhs.n), saved);
  EXPECT_EQ(std::vector<Value>(lu.val, lu.val + lu.size()), original);
}

TEST(BackwardSolve, NormalInput)
{
  const auto factors = MakeDenseFactors();
  const auto& lu = factors.lu;
  const std::vector<Value> original(lu.val, lu.val + lu.size());
  Vector rhs(lu.n);
  // Form U*x for x = [1, 2, ..., 8] independently of ForwardSolve.
  for (Index row = 0; row < lu.n; ++row) {
    rhs[row] = 0;
    for (Index col = row; col < lu.n; ++col) rhs[row] += lu(row, col) * (col + 1);
  }
  const std::vector<Value> saved(rhs.val, rhs.val + rhs.n);
  Vector x(lu.n);
  dense::BackwardSolve(lu, rhs, x);
  for (Index row = 0; row < lu.n; ++row) EXPECT_NEAR(x[row], row + 1, kTolerance);
  EXPECT_EQ(std::vector<Value>(rhs.val, rhs.val + rhs.n), saved);
  EXPECT_EQ(std::vector<Value>(lu.val, lu.val + lu.size()), original);
}

TEST(BackwardSolve, ZeroDiagonal)
{
  const DenseTestMatrix lu(1, 1, Value{0});
  const Vector rhs(1, 1);
  Vector x(1);
  EXPECT_THROW(dense::BackwardSolve(lu, rhs, x), std::runtime_error);
}

TEST(BackwardSolve, InfiniteDiagonal)
{
  const DenseTestMatrix lu(1, 1, std::numeric_limits<Value>::infinity());
  const Vector rhs(1, 1);
  Vector x(1);
  EXPECT_THROW(dense::BackwardSolve(lu, rhs, x), std::runtime_error);
}

namespace
{

using DenseFactorize = dense::Factors<Index, Value> (*)(const DenseTestMatrix&, const LuOptions<Value>&);

void CheckPackedFactorsAndPreservedInput(DenseFactorize factorize)
{
  const auto expected = MakeDenseFactors();
  const auto A = MakeDenseExample(expected.lu);
  const Index n = A.n;
  const std::vector<Value> original(A.val, A.val + A.size());
  auto factors = factorize(A, {});
  ASSERT_EQ(factors.status.code, LuCode::Success);
  EXPECT_EQ(factors.status.pivot_column, -1);
  ASSERT_EQ(factors.lu.m, n);
  ASSERT_EQ(factors.lu.n, n);
  EXPECT_TRUE(factors.lu.memowned);
  EXPECT_NE(factors.lu.val, A.val);
  for (Index col = 0; col < n; ++col) {
    for (Index row = 0; row < n; ++row) {
      SCOPED_TRACE(::testing::Message() << "row " << row << ", column " << col);
      EXPECT_NEAR(factors.lu(row, col), expected.lu(row, col), kTolerance);
      EXPECT_NEAR(PackedProductEntry(factors.lu, row, col), A(row, col), kTolerance);
    }
  }
  EXPECT_EQ(std::vector<Value>(A.val, A.val + A.size()), original);
  factors.lu(0, 0) = 99;
  EXPECT_EQ(A(0, 0), original.front());
}

void CheckZeroPivotAfterUpdate(DenseFactorize factorize)
{
  Value values[] = {2, 1, 4, 2};
  const DenseTestMatrix A(values, 2, 2);
  const auto factors = factorize(A, {});
  EXPECT_EQ(factors.status.code, LuCode::ZeroPivot);
  EXPECT_EQ(factors.status.pivot_column, 1);
}

void CheckInfinitePivotAfterUpdate(DenseFactorize factorize)
{
  // The second pivot overflows when subtracting 2 * max from 1.
  Value values[] = {1, 2, std::numeric_limits<Value>::max(), 1};
  const DenseTestMatrix A(values, 2, 2);
  const auto factors = factorize(A, {});
  EXPECT_EQ(factors.status.code, LuCode::ZeroPivot);
  EXPECT_EQ(factors.status.pivot_column, 1);
}

}  // namespace

TEST(RightLookingFactorize, NormalInput)
{
  CheckPackedFactorsAndPreservedInput(dense::RightLookingFactorize);
}

TEST(RightLookingFactorize, ZeroPivotAfterUpdate) { CheckZeroPivotAfterUpdate(dense::RightLookingFactorize); }

TEST(RightLookingFactorize, InfinitePivotAfterUpdate)
{
  CheckInfinitePivotAfterUpdate(dense::RightLookingFactorize);
}

TEST(LeftLookingFactorize, NormalInput) { CheckPackedFactorsAndPreservedInput(dense::LeftLookingFactorize); }

TEST(LeftLookingFactorize, ZeroPivotAfterUpdate) { CheckZeroPivotAfterUpdate(dense::LeftLookingFactorize); }

TEST(LeftLookingFactorize, InfinitePivotAfterUpdate)
{
  CheckInfinitePivotAfterUpdate(dense::LeftLookingFactorize);
}

TEST(UpperLookingFactorize, NormalInput)
{
  CheckPackedFactorsAndPreservedInput(dense::UpperLookingFactorize);
}

TEST(Solve, KnownFactorsReusableRhsAndOriginalResidual)
{
  // 1. Solve with supplied factors so Solve does not depend on RightLookingFactorize.
  const auto factors = MakeDenseFactors();
  const auto& lu = factors.lu;
  const std::vector<Value> original(lu.val, lu.val + lu.size());
  const auto A = MakeDenseExample(lu);
  Vector x(A.n);
  for (Value scale : {1, 2}) {
    const auto rhs = MakeDenseRhs(A, scale);
    const std::vector<Value> saved(rhs.val, rhs.val + rhs.n);
    dense::Solve(factors, rhs, x);
    for (Index row = 0; row < A.m; ++row) {
      EXPECT_NEAR(x[row], scale * (row + 1), kTolerance);
      Value product = 0;
      for (Index col = 0; col < A.n; ++col) product += A(row, col) * x[col];
      EXPECT_NEAR(product, rhs[row], kTolerance);
    }
    EXPECT_EQ(std::vector<Value>(rhs.val, rhs.val + rhs.n), saved);
  }
  EXPECT_EQ(std::vector<Value>(lu.val, lu.val + lu.size()), original);
  EXPECT_EQ(factors.status.code, LuCode::Success);
}

TEST(Solve, Empty)
{
  // 1. Successful empty factors require no substitution work.
  dense::Factors<Index, Value> factors;
  Vector empty;
  Vector x;
  factors.status.code = LuCode::Success;
  EXPECT_NO_THROW(dense::Solve(factors, empty, x));
}

TEST(Integration, RightLookingFactorizeThenSolve)
{
  // 1. Integrate the dense driver only after all four isolated checkpoints work.
  const auto expected = MakeDenseFactors();
  const auto A = MakeDenseExample(expected.lu);
  const auto factors = dense::RightLookingFactorize(A);
  ASSERT_EQ(factors.status.code, LuCode::Success);
  const auto rhs = MakeDenseRhs(A, 1);
  Vector x(A.n);
  dense::Solve(factors, rhs, x);
  for (Index i = 0; i < x.n; ++i) EXPECT_NEAR(x[i], i + 1, kTolerance);
}

}  // namespace spcraft::solver::test
