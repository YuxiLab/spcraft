#include "LuFixture.h"

namespace spcraft::solver::test
{

TEST(ScalarWorkspace, ResizeInitializesAndResetsScratch)
{
  scalar::Workspace<Index, Value> workspace;
  for (Index n : {4, 4, 7, 2, 0}) {
    workspace.Resize(n);
    ASSERT_EQ(workspace.values.size(), n);
    ASSERT_EQ(workspace.inverse_order.size(), n);
    ASSERT_EQ(workspace.marked.size(), n);
    EXPECT_TRUE(workspace.values.memowned);
    EXPECT_TRUE(workspace.touched.empty());
    EXPECT_GE(workspace.touched.capacity(), n);
    for (Index i = 0; i < n; ++i) {
      EXPECT_EQ(workspace.values[i], Value{0});
      EXPECT_EQ(workspace.inverse_order[i], i);
      EXPECT_EQ(workspace.marked[i], 0);
      workspace.values[i] = Value{3};
      workspace.inverse_order[i] = n - i - 1;
      workspace.marked[i] = 1;
      workspace.touched.push_back(i);
    }
  }
  EXPECT_THROW(workspace.Resize(Index{-1}), std::invalid_argument);
  EXPECT_EQ(workspace.values.size(), 0);
  EXPECT_TRUE(workspace.touched.empty());
}

TEST(ScalarLu, FillPermutationAndReusableSolve)
{
  for (bool permuted : {false, true}) {
    auto A = MakeMatrix();
    const auto original = Dense(A);
    const auto plan = MakeScalarPlan(permuted);
    auto factors = scalar::Factorize(A, plan);
    ASSERT_EQ(factors.status.code, LuCode::Success);
    EXPECT_EQ(factors.status.pivot_column, -1);
    CheckProduct(A, factors.order, Dense(factors.lower), Dense(factors.upper));
    EXPECT_EQ(Dense(A), original);
    auto b = MakeRhs();
    Vector x(4);
    scalar::Solve(factors, b, x);
    CheckSolution(x);
    for (Index i = 0; i < b.n; ++i) b.val[i] *= 2;
    scalar::Solve(factors, b, x);
    CheckSolution(x, 2);
  }
}

TEST(ScalarLu, NonsymmetricMapsAndReusableSolve)
{
  // 1. Test both coordinate orders against the original nonsymmetric system.
  for (bool permuted : {false, true}) {
    const auto A = MakeNonsymmetricMatrix(permuted);
    const auto original = Dense(A);
    const auto plan = MakeNonsymmetricScalarPlan(permuted);
    const auto factors = scalar::Factorize(A, plan);
    ASSERT_EQ(factors.status.code, LuCode::Success);
    CheckProduct(A, plan.order, Dense(factors.lower), Dense(factors.upper));
    auto b = MakeRhs(A);
    Vector x(4);
    const std::vector<Value> rhs(b.val, b.val + 4);
    scalar::Solve(factors, b, x);
    CheckSolution(x);
    EXPECT_EQ(std::vector<Value>(b.val, b.val + 4), rhs);
    for (Index i = 0; i < 4; ++i) b[i] *= 2;
    scalar::Solve(factors, b, x);
    CheckSolution(x, 2);
    EXPECT_EQ(Dense(A), original);
  }
}

TEST(ScalarLu, RejectsZeroPivotAndUnsuccessfulSolve)
{
  auto A = MakeZeroPivotMatrix();
  auto factors = scalar::Factorize(A, MakeZeroPivotPlan());
  ASSERT_EQ(factors.status.code, LuCode::ZeroPivot);
  EXPECT_EQ(factors.status.pivot_column, 0);
  Vector b(2, 1);
  Vector x(2);
  EXPECT_THROW(scalar::Solve(factors, b, x), std::invalid_argument);
}

TEST(ScalarLu, EmptyProblemAndInvalidInput)
{
  Matrix A;
  A.Allocate(0, 0, 0);
  LuSymbolic<Index> plan;
  plan.lower_col_ptr = {0};
  plan.upper_col_ptr = {0};
  auto factors = scalar::Factorize(A, plan);
  ASSERT_EQ(factors.status.code, LuCode::Success);
  Vector b;
  Vector x;
  EXPECT_NO_THROW(scalar::Solve(factors, b, x));
  auto nonempty = MakeMatrix();
  auto invalid = MakeScalarPlan();
  invalid.order = {0, 0, 2, 3};
  EXPECT_THROW(scalar::Factorize(nonempty, invalid), std::invalid_argument);
  EXPECT_THROW(scalar::Factorize(nonempty, MakeScalarPlan(), LuOptions<Value>{-1}), std::invalid_argument);
}

}  // namespace spcraft::solver::test
