#include "LuFixture.h"

namespace spcraft::solver::test
{

TEST(W21AssemblePanel, DifferentMapsPermutationAndWorkspaceReuse)
{
  for (bool permuted : {false, true}) {
    // 1. Gather two panels into the same workspace to detect stale maps or values.
    const auto A = MakeNonsymmetricMatrix(permuted);
    const auto original = Dense(A);
    const auto plan = MakeNonsymmetricBlockPlan(permuted);
    supernodal::Workspace<Index, Value> workspace;
    supernodal::AssemblePanel(A, plan, Index{0}, workspace);
    ASSERT_EQ(workspace.panel.diagonal.size(), 1);
    ASSERT_EQ(workspace.panel.lower.size(), 2);
    ASSERT_EQ(workspace.panel.upper.size(), 1);
    EXPECT_EQ(workspace.panel.diagonal.m, 1);
    EXPECT_EQ(workspace.panel.diagonal.n, 1);
    EXPECT_EQ(workspace.panel.lower.m, 2);
    EXPECT_EQ(workspace.panel.lower.n, 1);
    EXPECT_EQ(workspace.panel.upper.m, 1);
    EXPECT_EQ(workspace.panel.upper.n, 1);
    EXPECT_EQ(workspace.panel.rows, (std::vector<Index>{1, 3}));
    EXPECT_EQ(workspace.panel.columns, (std::vector<Index>{2}));
    EXPECT_EQ(workspace.panel.diagonal.val[0], 8);
    EXPECT_EQ(workspace.panel.lower.val[0], 2);
    EXPECT_EQ(workspace.panel.lower.val[1], 4);
    EXPECT_EQ(workspace.panel.upper.val[0], 2);
    supernodal::AssemblePanel(A, plan, Index{1}, workspace);
    ASSERT_EQ(workspace.panel.diagonal.size(), 1);
    ASSERT_EQ(workspace.panel.lower.size(), 1);
    ASSERT_EQ(workspace.panel.upper.size(), 2);
    EXPECT_EQ(workspace.panel.lower.m, 1);
    EXPECT_EQ(workspace.panel.lower.n, 1);
    EXPECT_EQ(workspace.panel.upper.m, 1);
    EXPECT_EQ(workspace.panel.upper.n, 2);
    EXPECT_EQ(workspace.panel.first, 1);
    EXPECT_EQ(workspace.panel.last, 2);
    EXPECT_EQ(workspace.panel.rows, (std::vector<Index>{2}));
    EXPECT_EQ(workspace.panel.columns, (std::vector<Index>{2, 3}));
    EXPECT_EQ(workspace.panel.diagonal.val[0], 10);
    EXPECT_EQ(workspace.panel.lower.val[0], 2);
    EXPECT_EQ(workspace.panel.upper.val[0], 0.5);
    EXPECT_EQ(workspace.panel.upper.val[1], 3);
    // 2. A wider panel replaces the old off-diagonal maps and buffers.
    supernodal::AssemblePanel(A, plan, Index{2}, workspace);
    ASSERT_EQ(workspace.panel.diagonal.size(), 4);
    EXPECT_EQ(workspace.panel.lower.size(), 0);
    EXPECT_EQ(workspace.panel.upper.size(), 0);
    EXPECT_EQ(workspace.panel.diagonal.m, 2);
    EXPECT_EQ(workspace.panel.diagonal.n, 2);
    EXPECT_EQ(workspace.panel.lower.m, 0);
    EXPECT_EQ(workspace.panel.lower.n, 2);
    EXPECT_EQ(workspace.panel.upper.m, 2);
    EXPECT_EQ(workspace.panel.upper.n, 0);
    EXPECT_EQ(workspace.panel.diagonal.val[0], 12);
    EXPECT_EQ(workspace.panel.diagonal.val[1], 4);
    EXPECT_EQ(workspace.panel.diagonal.val[2], 1.6);
    EXPECT_EQ(workspace.panel.diagonal.val[3], 16.25);
    // 3. A full-width cross includes absent input entries that must become zeros.
    auto full_plan = plan;
    full_plan.panels = {{0, 4, {}, {}, {}}};
    supernodal::AssemblePanel(A, full_plan, Index{0}, workspace);
    ASSERT_EQ(workspace.panel.diagonal.size(), 16);
    EXPECT_EQ(workspace.panel.diagonal.val[2], 0);
    EXPECT_EQ(workspace.panel.diagonal.val[4], 0);
    EXPECT_EQ(workspace.panel.diagonal.val[7], 0);
    EXPECT_EQ(workspace.panel.diagonal.val[12], 0);
    EXPECT_EQ(Dense(A), original);
  }
}

TEST(W22UpdatePanel, AllThreeProductsUseGlobalSourceMaps)
{
  // 1. Supply one completed width-two source with a missing row-3 coupling.
  SupernodalSymbolic<Index> plan;
  plan.scalar.n = 6;
  plan.panels = {{0, 2, {2, 4, 5}, {2, 3, 5}, {}}, {2, 4, {4, 5}, {5}, {0}}, {4, 6, {}, {}, {0, 1}}};
  supernodal::Factors<Index, Value> factors;
  factors.n = 6;
  factors.panels.resize(3);
  auto& source = factors.panels[0];
  source.first = 0;
  source.last = 2;
  source.rows = {2, 4, 5};
  source.columns = {2, 3, 5};
  source.diagonal = DenseMatrix<Index, Value>(2, 2, Value{0});
  source.diagonal.val[0] = 4;
  source.diagonal.val[1] = 0.25;
  source.diagonal.val[2] = 2;
  source.diagonal.val[3] = 3;
  source.lower.Allocate(3, 2);
  source.upper.Allocate(2, 3);
  const Value lower[] = {1, 3, 5, 2, 4, 6};
  const Value upper[] = {2, 1, 4, 3, 6, 5};
  std::copy(std::begin(lower), std::end(lower), source.lower.val);
  std::copy(std::begin(upper), std::end(upper), source.upper.val);
  // 2. Start with a known cross; this test does not require AssemblePanel.
  supernodal::Workspace<Index, Value> workspace;
  workspace.panel.first = 2;
  workspace.panel.last = 4;
  workspace.panel.rows = {4, 5};
  workspace.panel.columns = {5};
  workspace.panel.diagonal = DenseMatrix<Index, Value>(2, 2, Value{100});
  workspace.panel.lower = DenseMatrix<Index, Value>(2, 2, Value{100});
  workspace.panel.upper = DenseMatrix<Index, Value>(2, 1, Value{100});
  supernodal::UpdatePanel(plan, Index{1}, factors, workspace);
  // 3. Compare three independently hand-multiplied products in column-major order.
  const Value diagonal_expected[] = {96, 100, 90, 100};
  const Value lower_expected[] = {90, 84, 76, 62};
  const Value upper_expected[] = {84, 100};
  for (Index i = 0; i < 4; ++i) {
    EXPECT_NEAR(workspace.panel.diagonal.val[i], diagonal_expected[i], kTolerance);
    EXPECT_NEAR(workspace.panel.lower.val[i], lower_expected[i], kTolerance);
  }
  for (Index i = 0; i < 2; ++i) {
    EXPECT_NEAR(workspace.panel.upper.val[i], upper_expected[i], kTolerance);
  }
  for (Index i = 0; i < 6; ++i) {
    EXPECT_EQ(source.lower.val[i], lower[i]);
    EXPECT_EQ(source.upper.val[i], upper[i]);
  }
  EXPECT_EQ(workspace.panel.rows, (std::vector<Index>{4, 5}));
  EXPECT_EQ(workspace.panel.columns, (std::vector<Index>{5}));
}

TEST(W22UpdatePanel, AccumulatesTwoContributors)
{
  // 1. Provide the two earlier factored panels from the nonsymmetric example.
  const auto plan = MakeNonsymmetricBlockPlan();
  supernodal::Factors<Index, Value> factors;
  factors.n = 4;
  factors.panels.resize(3);
  for (Index s = 0; s < 2; ++s) {
    auto& source = factors.panels[s];
    source.first = plan.panels[s].first;
    source.last = plan.panels[s].last;
    source.rows = plan.panels[s].rows;
    source.columns = plan.panels[s].columns;
    source.diagonal = DenseMatrix<Index, Value>(1, 1, s == 0 ? 8 : 10);
    source.lower.Allocate(static_cast<Index>(source.rows.size()), 1);
    source.upper.Allocate(1, static_cast<Index>(source.columns.size()));
  }
  factors.panels[0].lower.val[0] = 0.25;
  factors.panels[0].lower.val[1] = 0.5;
  factors.panels[0].upper.val[0] = 2;
  factors.panels[1].lower.val[0] = 0.2;
  factors.panels[1].upper.val[0] = 0;
  factors.panels[1].upper.val[1] = 3;
  // 2. Subtract both products from panel 2's original diagonal cross.
  supernodal::Workspace<Index, Value> workspace;
  workspace.panel.first = 2;
  workspace.panel.last = 4;
  workspace.panel.lower.Allocate(0, 2);
  workspace.panel.upper.Allocate(2, 0);
  workspace.panel.diagonal.Allocate(2, 2);
  const Value original[] = {12, 4, 1.6, 16.25};
  std::copy(std::begin(original), std::end(original), workspace.panel.diagonal.val);
  supernodal::UpdatePanel(plan, Index{2}, factors, workspace);
  const Value expected[] = {12, 3, 1, 16.25};
  for (Index i = 0; i < 4; ++i) {
    EXPECT_NEAR(workspace.panel.diagonal.val[i], expected[i], kTolerance);
  }
}

TEST(W23FactorPanel, PackedLuAndBothTriangularSolves)
{
  // 1. Supply the updated cross directly, independent of assembly and updates.
  supernodal::Workspace<Index, Value> workspace;
  workspace.panel.first = 2;
  workspace.panel.last = 4;
  workspace.panel.rows = {4};
  workspace.panel.columns = {5};
  workspace.panel.diagonal.Allocate(2, 2);
  workspace.panel.lower.Allocate(1, 2);
  workspace.panel.upper.Allocate(2, 1);
  const Value diagonal[] = {4, 2, 2, 4};
  std::copy(std::begin(diagonal), std::end(diagonal), workspace.panel.diagonal.val);
  workspace.panel.lower.val[0] = 4;
  workspace.panel.lower.val[1] = 8;
  workspace.panel.upper.val[0] = 2;
  workspace.panel.upper.val[1] = 2;
  supernodal::Panel<Index, Value> output;
  const auto status = supernodal::FactorPanel(workspace, output, {});
  // 2. Check the packed triangles and the right/left solve results separately.
  ASSERT_EQ(status.code, LuCode::Success);
  EXPECT_EQ(status.pivot_column, -1);
  ASSERT_EQ(output.diagonal.size(), 4);
  ASSERT_EQ(output.lower.size(), 2);
  ASSERT_EQ(output.upper.size(), 2);
  const Value expected[] = {4, 0.5, 2, 3};
  for (Index i = 0; i < 4; ++i) EXPECT_NEAR(output.diagonal.val[i], expected[i], kTolerance);
  EXPECT_NEAR(output.lower.val[0], 1, kTolerance);
  EXPECT_NEAR(output.lower.val[1], 2, kTolerance);
  EXPECT_NEAR(output.upper.val[0], 2, kTolerance);
  EXPECT_NEAR(output.upper.val[1], 1, kTolerance);
  EXPECT_EQ(output.first, 2);
  EXPECT_EQ(output.last, 4);
  EXPECT_EQ(output.rows, (std::vector<Index>{4}));
  EXPECT_EQ(output.columns, (std::vector<Index>{5}));
  EXPECT_TRUE(output.diagonal.memowned);
}

TEST(W23FactorPanel, LaterPivotReportsGlobalColumn)
{
  // 1. The first pivot is valid, but the updated second pivot becomes zero.
  supernodal::Workspace<Index, Value> workspace;
  workspace.panel.first = 5;
  workspace.panel.last = 7;
  workspace.panel.lower.Allocate(0, 2);
  workspace.panel.upper.Allocate(2, 0);
  workspace.panel.diagonal.Allocate(2, 2);
  const Value diagonal[] = {2, 1, 4, 2};
  std::copy(std::begin(diagonal), std::end(diagonal), workspace.panel.diagonal.val);
  supernodal::Panel<Index, Value> output;
  const auto status = supernodal::FactorPanel(workspace, output, {});
  EXPECT_EQ(status.code, LuCode::ZeroPivot);
  EXPECT_EQ(status.pivot_column, 6);
}

}  // namespace spcraft::solver::test
