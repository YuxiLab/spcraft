#include "LuFixture.h"

#include <limits>

using namespace spcraft;
using namespace spcraft::solver;
using namespace spcraft::solver::test;

Matrix MakeSparseStarMatrix()
{
  // Supply the guide's four-by-four input, with absent entries below column 1's diagonal.
  Matrix A;
  A.Allocate(10, 4, 4);
  const Index pointers[] = {0, 4, 6, 8, 10};
  const Index rows[] = {0, 1, 2, 3, 0, 1, 0, 2, 0, 3};
  const Value values[] = {4, 1, 1, 1, 1, 4, 1, 4, 1, 4};
  std::copy(std::begin(pointers), std::end(pointers), A.col_ptr);
  std::copy(std::begin(rows), std::end(rows), A.row_id);
  std::copy(std::begin(values), std::end(values), A.val);
  return A;
}

scalar::Factors<Index, Value> MakeSparseStarFactors()
{
  // Supply independent hand-calculated factors so phase tests need no numerical driver.
  scalar::Factors<Index, Value> factors;
  factors.lower.Allocate(10, 4, 4);
  factors.upper.Allocate(10, 4, 4);
  const Index lower_pointers[] = {0, 4, 7, 9, 10};
  const Index lower_rows[] = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
  const Value lower_values[] = {1, 0.25, 0.25, 0.25, 1, -1.0 / 15, -1.0 / 15, 1, -1.0 / 14, 1};
  const Index upper_pointers[] = {0, 1, 3, 6, 10};
  const Index upper_rows[] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
  const Value upper_values[] = {4, 1, 15.0 / 4, 1, -0.25, 56.0 / 15, 1, -0.25, -4.0 / 15, 26.0 / 7};
  std::copy(std::begin(lower_pointers), std::end(lower_pointers), factors.lower.col_ptr);
  std::copy(std::begin(lower_rows), std::end(lower_rows), factors.lower.row_id);
  std::copy(std::begin(lower_values), std::end(lower_values), factors.lower.val);
  std::copy(std::begin(upper_pointers), std::end(upper_pointers), factors.upper.col_ptr);
  std::copy(std::begin(upper_rows), std::end(upper_rows), factors.upper.row_id);
  std::copy(std::begin(upper_values), std::end(upper_values), factors.upper.val);
  return factors;
}

TEST(SparseColumnFixtures, KnownFactorsReconstructInput)
{
  const Matrix A = MakeSparseStarMatrix();
  const scalar::Factors<Index, Value> factors = MakeSparseStarFactors();
  CheckProduct(A, {}, Dense(factors.lower), Dense(factors.upper));
}

TEST(SparseColumnWorkspace, ClearPreservesPermutationAndAllocation)
{
  scalar::Workspace<Index, Value> workspace;
  workspace.Resize(4);
  workspace.inverse_order = {1, 3, 0, 2};
  workspace.values[0] = 3;
  workspace.values[2] = -5;
  workspace.marked[0] = workspace.marked[2] = 1;
  workspace.touched = {2, 0};
  Value* const buffer = workspace.values.val;
  const std::size_t capacity = workspace.touched.capacity();
  workspace.ClearColumn();
  EXPECT_EQ(workspace.inverse_order, (std::vector<Index>{1, 3, 0, 2}));
  EXPECT_EQ(workspace.values.val, buffer);
  EXPECT_EQ(workspace.touched.capacity(), capacity);
  EXPECT_TRUE(workspace.touched.empty());
  for (Index row = 0; row < 4; ++row) {
    EXPECT_EQ(workspace.values[row], 0);
    EXPECT_EQ(workspace.marked[row], 0);
  }
  EXPECT_NO_THROW(workspace.ClearColumn());
}

TEST(SparseColumnWorkspace, EmptyCleanup)
{
  scalar::Workspace<Index, Value> workspace;
  workspace.Resize(0);
  EXPECT_NO_THROW(workspace.ClearColumn());
  EXPECT_TRUE(workspace.touched.empty());
  EXPECT_TRUE(workspace.inverse_order.empty());
}

TEST(SparseScatterColumn, UsesOriginalColumnAndOrderedRows)
{
  const Matrix A = MakeSparseStarMatrix();
  const std::vector<Value> original = Dense(A);
  scalar::Workspace<Index, Value> workspace;
  workspace.Resize(4);
  workspace.inverse_order = {1, 3, 0, 2};
  scalar::ScatterColumn(A, Index{1}, workspace);
  const Value expected[] = {0, 1, 0, 4};
  for (Index row = 0; row < 4; ++row) EXPECT_EQ(workspace.values[row], expected[row]);
  EXPECT_EQ(workspace.touched, (std::vector<Index>{1, 3}));
  EXPECT_EQ(workspace.marked, (std::vector<unsigned char>{0, 1, 0, 1}));
  EXPECT_EQ(workspace.inverse_order, (std::vector<Index>{1, 3, 0, 2}));
  EXPECT_EQ(Dense(A), original);
}

TEST(SparseUpdateColumn, CreatesFillWithoutReadingUnfinishedFactorValues)
{
  scalar::Factors<Index, Value> factors = MakeSparseStarFactors();
  std::fill(factors.lower.val + 4, factors.lower.val + 10, std::numeric_limits<Value>::quiet_NaN());
  std::fill(factors.upper.val + 1, factors.upper.val + 10, std::numeric_limits<Value>::quiet_NaN());
  scalar::Workspace<Index, Value> workspace;
  workspace.Resize(4);
  workspace.values[0] = 1;
  workspace.values[1] = 4;
  workspace.marked[0] = workspace.marked[1] = 1;
  workspace.touched = {0, 1};
  scalar::UpdateColumn(Index{1}, factors, workspace);
  const Value expected[] = {1, 15.0 / 4, -0.25, -0.25};
  for (Index row = 0; row < 4; ++row) {
    EXPECT_NEAR(workspace.values[row], expected[row], kTolerance);
    EXPECT_EQ(workspace.marked[row], 1);
  }
  std::vector<Index> touched = workspace.touched;
  std::sort(touched.begin(), touched.end());
  EXPECT_EQ(touched, (std::vector<Index>{0, 1, 2, 3}));
  EXPECT_TRUE(std::isnan(factors.upper.val[1]));
  EXPECT_TRUE(std::isnan(factors.lower.val[4]));
}

TEST(SparseStoreColumn, WritesKnownColumnAndPreservesScratch)
{
  scalar::Factors<Index, Value> factors = MakeSparseStarFactors();
  std::fill(factors.lower.val + 4, factors.lower.val + 7, Value{-99});
  std::fill(factors.upper.val + 1, factors.upper.val + 3, Value{-99});
  scalar::Workspace<Index, Value> workspace;
  workspace.Resize(4);
  const Value updated[] = {1, 15.0 / 4, -0.25, -0.25};
  std::copy(std::begin(updated), std::end(updated), workspace.values.val);
  const LuStatus<Index> status = scalar::StoreColumn(Index{1}, factors, workspace);
  ASSERT_EQ(status.code, LuCode::Success);
  EXPECT_EQ(status.pivot_column, -1);
  EXPECT_EQ(factors.status.code, LuCode::NotFactorized);
  EXPECT_EQ(factors.upper.val[1], 1);
  EXPECT_EQ(factors.upper.val[2], 15.0 / 4);
  EXPECT_EQ(factors.lower.val[4], 1);
  EXPECT_NEAR(factors.lower.val[5], -1.0 / 15, kTolerance);
  EXPECT_NEAR(factors.lower.val[6], -1.0 / 15, kTolerance);
  for (Index row = 0; row < 4; ++row) EXPECT_EQ(workspace.values[row], updated[row]);
  workspace.values[0] = 0;
  factors.upper.val[1] = -99;
  const LuStatus<Index> zero_status = scalar::StoreColumn(Index{1}, factors, workspace);
  EXPECT_EQ(zero_status.code, LuCode::Success);
  EXPECT_EQ(factors.upper.val[1], 0);
  EXPECT_EQ(factors.upper.row_id[1], 0);
}

TEST(SparseStoreColumn, RejectsZeroNonfiniteAndToleranceBoundaryPivots)
{
  scalar::Factors<Index, Value> factors = MakeSparseStarFactors();
  scalar::Workspace<Index, Value> workspace;
  workspace.Resize(4);
  for (Value pivot : {Value{0}, std::numeric_limits<Value>::infinity()}) {
    workspace.values[1] = pivot;
    const LuStatus<Index> status = scalar::StoreColumn(Index{1}, factors, workspace);
    EXPECT_EQ(status.code, LuCode::ZeroPivot);
    EXPECT_EQ(status.pivot_column, 1);
  }
  workspace.values[1] = 15.0 / 4;
  const LuOptions<Value> options{15.0 / 4};
  const LuStatus<Index> status = scalar::StoreColumn(Index{1}, factors, workspace, options);
  EXPECT_EQ(status.code, LuCode::ZeroPivot);
  EXPECT_EQ(status.pivot_column, 1);
}
