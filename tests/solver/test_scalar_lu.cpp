#include "LuFixture.h"

namespace spcraft::solver::test
{

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
