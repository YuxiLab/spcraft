#include "LuFixture.h"

namespace spcraft::solver::test
{

TEST(SupernodalLu, GenuineBlocksAndPermutedSolve)
{
  auto A = MakeMatrix();
  const auto original = Dense(A);
  const auto plan = MakeBlockPlan();
  auto factors = supernodal::Factorize(A, plan);
  ASSERT_EQ(factors.status.code, LuCode::Success);
  ASSERT_EQ(factors.panels.size(), plan.panels.size());
  CheckBlocks(A, factors);
  EXPECT_EQ(Dense(A), original);
  auto b = MakeRhs();
  Vector x(4);
  supernodal::Solve(factors, b, x);
  CheckSolution(x);
  for (Index i = 0; i < b.n; ++i) b.val[i] *= 2;
  supernodal::Solve(factors, b, x);
  CheckSolution(x, 2);
}

TEST(SupernodalLu, NonsymmetricMapsAndReusableSolve)
{
  // 1. Check distinct lower/upper maps, a width-two panel, and both permutations.
  for (bool permuted : {false, true}) {
    const auto A = MakeNonsymmetricMatrix(permuted);
    const auto original = Dense(A);
    const auto plan = MakeNonsymmetricBlockPlan(permuted);
    const auto factors = supernodal::Factorize(A, plan);
    ASSERT_EQ(factors.status.code, LuCode::Success);
    CheckBlocks(A, factors);
    auto b = MakeRhs(A);
    Vector x(4);
    const std::vector<Value> rhs(b.val, b.val + 4);
    supernodal::Solve(factors, b, x);
    CheckSolution(x);
    EXPECT_EQ(std::vector<Value>(b.val, b.val + 4), rhs);
    for (Index i = 0; i < 4; ++i) b[i] *= 2;
    supernodal::Solve(factors, b, x);
    CheckSolution(x, 2);
    EXPECT_EQ(Dense(A), original);
  }
}

TEST(SupernodalLu, EmptyAndRejectedPivot)
{
  Matrix empty;
  empty.Allocate(0, 0, 0);
  SupernodalSymbolic<Index> plan;
  plan.scalar.lower_col_ptr = {0};
  plan.scalar.upper_col_ptr = {0};
  auto factors = supernodal::Factorize(empty, plan);
  ASSERT_EQ(factors.status.code, LuCode::Success);
  Vector b;
  Vector x;
  EXPECT_NO_THROW(supernodal::Solve(factors, b, x));
  auto A = MakeZeroPivotMatrix();
  plan.scalar = MakeZeroPivotPlan();
  plan.panels = {{0, 2, {}, {}, {}}};
  auto failed = supernodal::Factorize(A, plan);
  EXPECT_EQ(failed.status.code, LuCode::ZeroPivot);
  EXPECT_EQ(failed.status.pivot_column, 0);
}

}  // namespace spcraft::solver::test
