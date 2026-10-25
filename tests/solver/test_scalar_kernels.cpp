#include "LuFixture.h"

namespace spcraft::solver::test
{

TEST(LearningFixtures, NonsymmetricFactorsAndPermutation)
{
  // 1. Check hand-supplied factors independently of unfinished student kernels.
  const std::vector<Value> lower = {1, 0.25, 0, 0.5, 0, 1, 0.2, 0, 0, 0, 1, 0.25, 0, 0, 0, 1};
  const std::vector<Value> upper = {8, 0, 0, 0, 0, 10, 0, 0, 2, 0, 12, 0, 0, 3, 1, 16};
  for (bool permuted : {false, true}) {
    const auto A = MakeNonsymmetricMatrix(permuted);
    const auto plan = MakeNonsymmetricScalarPlan(permuted);
    CheckProduct(A, plan.order, lower, upper);
    EXPECT_EQ(A.col_ptr[4], 12);
    const auto dense = Dense(A);
    for (Index row = 0; row < 4; ++row) {
      Value off_diagonal = 0;
      for (Index col = 0; col < 4; ++col) {
        if (row != col) off_diagonal += std::abs(dense[row + col * 4]);
      }
      EXPECT_GT(std::abs(dense[row + row * 4]), off_diagonal);
    }
  }
}

TEST(W13ForwardSolve, KnownUnitLowerAndPreservedRhs)
{
  // 1. Supply L and b directly; no factorization function is needed for this test.
  Matrix lower;
  lower.Allocate(5, 3, 3);
  const Index ptr[] = {0, 2, 4, 5};
  const Index rows[] = {0, 1, 1, 2, 2};
  const Value values[] = {1, 0.5, 1, 2.0 / 9.0, 1};
  std::copy(std::begin(ptr), std::end(ptr), lower.col_ptr);
  std::copy(std::begin(rows), std::end(rows), lower.row_id);
  std::copy(std::begin(values), std::end(values), lower.val);
  Vector rhs(3);
  rhs[0] = 9;
  rhs[1] = 12;
  rhs[2] = 20;
  Vector y(3);
  scalar::ForwardSolve(lower, rhs, y);
  // 2. Compare against the hand-worked intermediate vector and unchanged inputs.
  EXPECT_NEAR(y[0], 9, kTolerance);
  EXPECT_NEAR(y[1], 7.5, kTolerance);
  EXPECT_NEAR(y[2], 55.0 / 3.0, kTolerance);
  EXPECT_EQ(rhs[0], 9);
  EXPECT_EQ(rhs[1], 12);
  EXPECT_EQ(rhs[2], 20);
  EXPECT_EQ(Dense(lower), (std::vector<Value>{1, 0.5, 0, 0, 1, 2.0 / 9.0, 0, 0, 1}));
}

TEST(W13ForwardSolve, EmptyAndNonunitDiagonal)
{
  // 1. An empty solve is valid; a nonunit lower diagonal violates this contract.
  Matrix lower;
  lower.Allocate(0, 0, 0);
  Vector rhs;
  Vector y;
  EXPECT_NO_THROW(scalar::ForwardSolve(lower, rhs, y));
  lower.Allocate(1, 1, 1);
  lower.col_ptr[0] = 0;
  lower.col_ptr[1] = 1;
  lower.row_id[0] = 0;
  lower.val[0] = 2;
  rhs.Allocate(1);
  y.Allocate(1);
  rhs[0] = 3;
  EXPECT_THROW(scalar::ForwardSolve(lower, rhs, y), std::runtime_error);
}

TEST(W14BackwardSolve, KnownUpperAndPreservedRhs)
{
  // 1. Supply U and the forward solve's expected y independently of W1.3.
  Matrix upper;
  upper.Allocate(6, 3, 3);
  const Index ptr[] = {0, 1, 3, 6};
  const Index rows[] = {0, 0, 1, 0, 1, 2};
  const Value values[] = {4, 1, 4.5, 1, -0.5, 55.0 / 9.0};
  std::copy(std::begin(ptr), std::end(ptr), upper.col_ptr);
  std::copy(std::begin(rows), std::end(rows), upper.row_id);
  std::copy(std::begin(values), std::end(values), upper.val);
  const auto original = Dense(upper);
  Vector rhs(3);
  rhs[0] = 9;
  rhs[1] = 7.5;
  rhs[2] = 55.0 / 3.0;
  Vector x(3);
  scalar::BackwardSolve(upper, rhs, x);
  // 2. Check reverse substitution, including the negative fill coefficient.
  for (Index i = 0; i < 3; ++i) EXPECT_NEAR(x[i], i + 1, kTolerance);
  EXPECT_EQ(rhs[0], 9);
  EXPECT_EQ(rhs[1], 7.5);
  EXPECT_EQ(rhs[2], 55.0 / 3.0);
  EXPECT_EQ(Dense(upper), original);
}

TEST(W14BackwardSolve, EmptyAndZeroDiagonal)
{
  // 1. Check reverse traversal at n=0, then rejection before division by zero.
  Matrix upper;
  upper.Allocate(0, 0, 0);
  Vector rhs;
  Vector x;
  EXPECT_NO_THROW(scalar::BackwardSolve(upper, rhs, x));
  upper.Allocate(1, 1, 1);
  upper.col_ptr[0] = 0;
  upper.col_ptr[1] = 1;
  upper.row_id[0] = 0;
  upper.val[0] = 0;
  rhs.Allocate(1);
  x.Allocate(1);
  rhs[0] = 1;
  EXPECT_THROW(scalar::BackwardSolve(upper, rhs, x), std::runtime_error);
}

TEST(W11FactorColumn, NonsymmetricColumnsPermutationAndScratchReuse)
{
  for (bool permuted : {false, true}) {
    // 1. Allocate the complete plan without calling the unfinished Factorize driver.
    const auto A = MakeNonsymmetricMatrix(permuted);
    const auto original = Dense(A);
    const auto plan = MakeNonsymmetricScalarPlan(permuted);
    scalar::Factors<Index, Value> factors;
    factors.order = plan.order;
    factors.lower.Allocate(static_cast<Index>(plan.lower_row_id.size()), 4, 4);
    factors.upper.Allocate(static_cast<Index>(plan.upper_row_id.size()), 4, 4);
    std::copy(plan.lower_col_ptr.begin(), plan.lower_col_ptr.end(), factors.lower.col_ptr);
    std::copy(plan.lower_row_id.begin(), plan.lower_row_id.end(), factors.lower.row_id);
    std::copy(plan.upper_col_ptr.begin(), plan.upper_col_ptr.end(), factors.upper.col_ptr);
    std::copy(plan.upper_row_id.begin(), plan.upper_row_id.end(), factors.upper.row_id);
    std::fill_n(factors.lower.val, plan.lower_row_id.size(), Value{0});
    std::fill_n(factors.upper.val, plan.upper_row_id.size(), Value{0});
    scalar::Workspace<Index, Value> workspace;
    workspace.Resize(4);
    if (permuted) {
      for (Index i = 0; i < 4; ++i) workspace.inverse_order[plan.order[i]] = i;
    }
    const auto inverse = workspace.inverse_order;
    // 2. Visit columns in order; each completed column must leave reusable scratch.
    for (Index j = 0; j < 4; ++j) {
      const auto status = scalar::FactorColumn(A, plan, j, factors, workspace, {});
      ASSERT_EQ(status.code, LuCode::Success);
      EXPECT_EQ(status.pivot_column, -1);
      EXPECT_EQ(workspace.inverse_order, inverse);
      EXPECT_TRUE(workspace.touched.empty());
      for (Index i = 0; i < 4; ++i) {
        EXPECT_EQ(workspace.values[i], 0);
        EXPECT_EQ(workspace.marked[i], 0);
      }
    }
    // 3. Check the product, retained cancellation slot, and input preservation.
    CheckProduct(A, plan.order, Dense(factors.lower), Dense(factors.upper));
    EXPECT_NEAR(factors.upper.val[3], 0, kTolerance);
    EXPECT_EQ(Dense(A), original);
    EXPECT_EQ(factors.status.code, LuCode::NotFactorized);
  }
}

}  // namespace spcraft::solver::test
