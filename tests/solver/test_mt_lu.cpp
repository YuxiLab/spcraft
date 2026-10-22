#include "LuFixture.h"

namespace spcraft::solver::test
{

TEST(MtLu, ContributorGraphAndInvalidEdges)
{
  auto plan = MakeBlockPlan();
  auto graph = mt::BuildTaskGraph(plan);
  EXPECT_EQ(graph.successor_ptr, (std::vector<Index>{0, 2, 3, 3}));
  EXPECT_EQ(graph.successor_id, (std::vector<Index>{1, 2, 2}));
  EXPECT_EQ(graph.indegree, (std::vector<Index>{0, 1, 2}));
  plan.panels[0].contributors = {1};
  EXPECT_THROW(mt::BuildTaskGraph(plan), std::invalid_argument);
}

TEST(MtLu, FactorizationAcrossThreadCounts)
{
  for (int threads : {1, 2, 4, 8}) {
    auto A = MakeMatrix();
    const auto plan = MakeBlockPlan();
    mt::Options<Value> options;
    options.threads = threads;
    auto factors = mt::Factorize(A, plan, options);
    ASSERT_EQ(factors.status.code, LuCode::Success);
    CheckBlocks(A, factors);
    auto b = MakeRhs();
    Vector x(4);
    supernodal::Solve(factors, b, x);
    CheckSolution(x);
  }
}

TEST(MtLu, IndependentBranchesJoinBeforeParent)
{
  Matrix A;
  A.Allocate(7, 3, 3);
  const Index ptr[] = {0, 2, 4, 7};
  const Index rows[] = {0, 2, 1, 2, 0, 1, 2};
  const Value values[] = {4, 1, 5, 1, 1, 1, 6};
  std::copy(std::begin(ptr), std::end(ptr), A.col_ptr);
  std::copy(std::begin(rows), std::end(rows), A.row_id);
  std::copy(std::begin(values), std::end(values), A.val);
  SupernodalSymbolic<Index> plan;
  plan.scalar.n = 3;
  plan.scalar.lower_col_ptr = {0, 2, 4, 5};
  plan.scalar.lower_row_id = {0, 2, 1, 2, 2};
  plan.scalar.upper_col_ptr = {0, 1, 2, 5};
  plan.scalar.upper_row_id = {0, 1, 0, 1, 2};
  plan.panels = {{0, 1, {2}, {2}, {}}, {1, 2, {2}, {2}, {}}, {2, 3, {}, {}, {0, 1}}};
  mt::Options<Value> options;
  options.threads = 4;
  auto factors = mt::Factorize(A, plan, options);
  ASSERT_EQ(factors.status.code, LuCode::Success);
  CheckBlocks(A, factors);
  Vector b(3);
  b.val[0] = 7;
  b.val[1] = 13;
  b.val[2] = 21;
  Vector x(3);
  supernodal::Solve(factors, b, x);
  for (Index i = 0; i < 3; ++i) EXPECT_NEAR(x.val[i], i + 1, kTolerance);
}

TEST(MtLu, RejectedPivotReturnsWithoutDeadlock)
{
  auto A = MakeZeroPivotMatrix();
  SupernodalSymbolic<Index> plan;
  plan.scalar = MakeZeroPivotPlan();
  plan.panels = {{0, 1, {1}, {1}, {}}, {1, 2, {}, {}, {0}}};
  mt::Options<Value> options;
  options.threads = 4;
  auto factors = mt::Factorize(A, plan, options);
  EXPECT_EQ(factors.status.code, LuCode::ZeroPivot);
  EXPECT_EQ(factors.status.pivot_column, 0);
  options.threads = 0;
  EXPECT_THROW(mt::Factorize(A, plan, options), std::invalid_argument);
}

}  // namespace spcraft::solver::test
