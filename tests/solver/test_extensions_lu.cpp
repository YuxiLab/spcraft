#include "LuFixture.h"
#include "solver/expert/Lu.h"

#include <limits>
#include <type_traits>

using namespace spcraft;
using namespace spcraft::solver;
using namespace spcraft::solver::test;

Matrix MakeExtensionInput()
{
  Matrix A;
  A.Allocate(7, 4, 4);
  const Index pointers[] = {0, 1, 3, 5, 7};
  const Index rows[] = {1, 0, 1, 0, 3, 2, 3};
  const Value values[] = {1, 2, 3, 1, 2, 4, 5};
  std::copy(std::begin(pointers), std::end(pointers), A.col_ptr);
  std::copy(std::begin(rows), std::end(rows), A.row_id);
  std::copy(std::begin(values), std::end(values), A.val);
  return A;
}

symbolic::Analysis<Index> MakeExtensionAnalysis()
{
  symbolic::Analysis<Index> analysis;
  analysis.input_pattern.n = 4;
  analysis.input_pattern.col_ptr = {0, 1, 3, 5, 7};
  analysis.input_pattern.row_id = {1, 0, 1, 0, 3, 2, 3};
  analysis.col_order = {2, 3, 0, 1};
  analysis.inverse_col_order = {2, 3, 0, 1};
  analysis.etree = {1, 3, 3, -1};
  analysis.postorder = {0, 1, 2, 3};
  return analysis;
}

pivoted::Factors<Index, Value> MakeExtensionFactors()
{
  // Independently supplied factors: the two block swaps require different row/column maps.
  pivoted::Factors<Index, Value> factors;
  factors.lower.Allocate(6, 4, 4);
  factors.upper.Allocate(6, 4, 4);
  const Index lower_pointers[] = {0, 2, 4, 5, 6};
  const Index lower_rows[] = {0, 3, 1, 3, 2, 3};
  const Value lower_values[] = {1, 0.5, 1, -0.625, 1, 1};
  std::copy(std::begin(lower_pointers), std::end(lower_pointers), factors.lower.col_ptr);
  std::copy(std::begin(lower_rows), std::end(lower_rows), factors.lower.row_id);
  std::copy(std::begin(lower_values), std::end(lower_values), factors.lower.val);
  const Index pointers[] = {0, 1, 3, 4, 6};
  const Index rows[] = {0, 0, 1, 2, 2, 3};
  const Value values[] = {2, 5, 4, 1, 3, 2};
  std::copy(std::begin(pointers), std::end(pointers), factors.upper.col_ptr);
  std::copy(std::begin(rows), std::end(rows), factors.upper.row_id);
  std::copy(std::begin(values), std::end(values), factors.upper.val);
  factors.col_order = {2, 3, 0, 1};
  factors.row_order = {3, 2, 1, 0};
  factors.status.code = LuCode::Success;
  return factors;
}

expert::State<Index, Value> MakeExtensionState()
{
  expert::State<Index, Value> state;
  state.analysis = MakeExtensionAnalysis();
  state.scaling.rows = {2, 3, 4, 5};
  state.scaling.columns = {0.5, 2, 1, 0.25};
  state.factors = MakeExtensionFactors();
  const Value scaled_values[] = {10, 6.25, 4, 1.5, 18, 8};
  std::copy(std::begin(scaled_values), std::end(scaled_values), state.factors.upper.val);
  state.factors.lower.val[1] = 0.2;
  state.factors.lower.val[3] = -0.3125;
  return state;
}

TEST(ExtensionFixtures, IndependentFactorsReconstructInput)
{
  const Matrix A = MakeExtensionInput();
  const pivoted::Factors<Index, Value> factors = MakeExtensionFactors();
  const std::vector<Value> original = Dense(A);
  const std::vector<Value> lower = Dense(factors.lower);
  const std::vector<Value> upper = Dense(factors.upper);
  for (Index j = 0; j < 4; ++j) {
    for (Index i = 0; i < 4; ++i) {
      Value product = 0;
      for (Index k = 0; k < 4; ++k) product += lower[i + k * 4] * upper[k + j * 4];
      EXPECT_DOUBLE_EQ(product, original[factors.row_order[i] + factors.col_order[j] * 4]);
    }
  }
}

TEST(SymbolicAnalysis, NaturalUserOrderTreeAndForestPostorder)
{
  const symbolic::Pattern<Index> pattern = MakeExtensionAnalysis().input_pattern;
  symbolic::Options<Index> options;
  EXPECT_EQ(symbolic::ComputeColumnOrder(pattern, options), (std::vector<Index>{0, 1, 2, 3}));
  options.ordering = symbolic::ColumnOrdering::User;
  options.user_order = {2, 3, 0, 1};
  const symbolic::Analysis<Index> analysis = symbolic::Analyze(pattern, options);
  EXPECT_EQ(analysis.col_order, options.user_order);
  EXPECT_EQ(analysis.inverse_col_order, (std::vector<Index>{2, 3, 0, 1}));
  EXPECT_EQ(analysis.etree, (std::vector<Index>{1, 3, 3, -1}));
  EXPECT_EQ(analysis.postorder, (std::vector<Index>{0, 1, 2, 3}));
  EXPECT_EQ(symbolic::Postorder(std::vector<Index>{2, 2, -1, -1}), (std::vector<Index>{0, 1, 2, 3}));
}

TEST(SymbolicAnalysis, DiscoversNoPivotStarFill)
{
  symbolic::Pattern<Index> pattern;
  pattern.n = 4;
  pattern.col_ptr = {0, 4, 6, 8, 10};
  pattern.row_id = {0, 1, 2, 3, 0, 1, 0, 2, 0, 3};
  const LuSymbolic<Index> plan = symbolic::AnalyzeNoPivotPattern(pattern, std::vector<Index>{0, 1, 2, 3});
  EXPECT_EQ(plan.lower_col_ptr, (std::vector<Index>{0, 4, 7, 9, 10}));
  EXPECT_EQ(plan.lower_row_id, (std::vector<Index>{0, 1, 2, 3, 1, 2, 3, 2, 3, 3}));
  EXPECT_EQ(plan.upper_col_ptr, (std::vector<Index>{0, 1, 3, 6, 10}));
  EXPECT_EQ(plan.upper_row_id, (std::vector<Index>{0, 0, 1, 0, 1, 2, 0, 1, 2, 3}));
}

TEST(DynamicFactorStorage, GrowthPreservesEntriesAndSizes)
{
  pivoted::Column<Index, Value> column;
  column.rows = {1, 3};
  column.values = {2, -5};
  pivoted::ReserveColumn(column, Index{17});
  EXPECT_GE(column.rows.capacity(), 17u);
  EXPECT_GE(column.values.capacity(), 17u);
  EXPECT_EQ(column.rows, (std::vector<Index>{1, 3}));
  EXPECT_EQ(column.values, (std::vector<Value>{2, -5}));
}

TEST(SymbolicSupernodes, ProducesClosedTwoPanelPlanForExistingCore)
{
  LuSymbolic<Index> plan;
  plan.n = 4;
  plan.order = {0, 1, 2, 3};
  plan.lower_col_ptr = {0, 4, 7, 9, 10};
  plan.lower_row_id = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
  plan.upper_col_ptr = {0, 1, 3, 6, 10};
  plan.upper_row_id = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
  symbolic::Options<Index> options;
  options.max_supernode_width = 2;
  const SupernodalSymbolic<Index> blocks = symbolic::DetectSupernodes(plan, options);
  ASSERT_EQ(blocks.panels.size(), 2u);
  EXPECT_EQ(blocks.scalar.lower_row_id, plan.lower_row_id);
  EXPECT_EQ(blocks.scalar.upper_row_id, plan.upper_row_id);
  EXPECT_EQ(blocks.panels[0].first, 0);
  EXPECT_EQ(blocks.panels[0].last, 2);
  EXPECT_EQ(blocks.panels[0].rows, (std::vector<Index>{2, 3}));
  EXPECT_EQ(blocks.panels[0].columns, (std::vector<Index>{2, 3}));
  EXPECT_TRUE(blocks.panels[0].contributors.empty());
  EXPECT_EQ(blocks.panels[1].first, 2);
  EXPECT_EQ(blocks.panels[1].last, 4);
  EXPECT_EQ(blocks.panels[1].contributors, (std::vector<Index>{0}));
}

TEST(PivotedColumn, ReachAndUpdateUseOriginalRowsAndCompletedFactors)
{
  const Matrix A = MakeExtensionInput();
  pivoted::Assembly<Index, Value> assembly;
  assembly.n = 4;
  assembly.col_order = {2, 3, 0, 1};
  assembly.row_order = {3, -1, -1, -1};
  assembly.inverse_row_order = {-1, -1, -1, 0};
  assembly.lower.resize(4);
  assembly.upper.resize(4);
  assembly.lower[0].rows = {3, 0};
  assembly.lower[0].values = {1, 0.5};
  pivoted::Workspace<Index, Value> work;
  work.values.assign(4, 0);
  work.marked.assign(4, 0);
  pivoted::ScatterColumn(A, Index{3}, work);
  work.reach = pivoted::ReachColumn(A, Index{1}, assembly);
  EXPECT_EQ(work.reach, (std::vector<Index>{0}));
  pivoted::UpdateColumn(Index{1}, assembly, work);
  EXPECT_EQ(work.values, (std::vector<Value>{-2.5, 0, 4, 5}));
  EXPECT_EQ(work.marked[0], 1);
  EXPECT_EQ(assembly.row_order, (std::vector<Index>{3, -1, -1, -1}));
}

TEST(DynamicFactorStorage, FinalizationMapsAndSortsEarlierLowerRows)
{
  pivoted::Assembly<Index, Value> assembly;
  assembly.n = 4;
  assembly.col_order = {2, 3, 0, 1};
  assembly.row_order = {3, 2, 1, 0};
  assembly.inverse_row_order = {3, 2, 1, 0};
  assembly.lower.resize(4);
  assembly.upper.resize(4);
  assembly.lower[0].rows = {0, 3};
  assembly.lower[0].values = {0.5, 1};
  assembly.lower[1].rows = {0, 2};
  assembly.lower[1].values = {-0.625, 1};
  assembly.lower[2].rows = {1};
  assembly.lower[2].values = {1};
  assembly.lower[3].rows = {0};
  assembly.lower[3].values = {1};
  assembly.upper[0] = {{0}, {2}};
  assembly.upper[1] = {{0, 1}, {5, 4}};
  assembly.upper[2] = {{2}, {1}};
  assembly.upper[3] = {{2, 3}, {3, 2}};
  assembly.status.code = LuCode::Success;
  const pivoted::Factors<Index, Value> factors = pivoted::FinalizeFactors(std::move(assembly));
  const pivoted::Factors<Index, Value> expected = MakeExtensionFactors();
  EXPECT_EQ(factors.status.code, LuCode::Success);
  EXPECT_EQ(factors.row_order, expected.row_order);
  EXPECT_EQ(factors.col_order, expected.col_order);
  EXPECT_EQ(Dense(factors.lower), Dense(expected.lower));
  EXPECT_EQ(Dense(factors.upper), Dense(expected.upper));
  ASSERT_EQ(factors.lower.nnz, 6);
  EXPECT_EQ(factors.lower.row_id[0], 0);
  EXPECT_EQ(factors.lower.row_id[1], 3);
  EXPECT_DOUBLE_EQ(factors.lower.val[0], 1);
  EXPECT_DOUBLE_EQ(factors.lower.val[1], 0.5);
}

TEST(ThresholdPivot, RelativePreferenceAndAbsoluteRejectionAreSeparate)
{
  pivoted::Assembly<Index, Value> assembly;
  assembly.n = 4;
  assembly.col_order = {0, 1, 2, 3};
  assembly.row_order = {-1, -1, -1, -1};
  assembly.inverse_row_order = {-1, -1, -1, -1};
  pivoted::Workspace<Index, Value> work;
  work.values = {0.2, 2, 0, 0};
  work.marked = {1, 1, 0, 0};
  work.touched = {0, 1};
  pivoted::Options<Index, Value> options;
  options.diagonal_threshold = 0.1;
  EXPECT_EQ(pivoted::SelectPivot(Index{0}, assembly, work, options).old_row, 0);
  options.diagonal_threshold = 1;
  EXPECT_EQ(pivoted::SelectPivot(Index{0}, assembly, work, options).old_row, 1);
  options.pivot_tolerance = 2;
  const pivoted::Pivot<Index, Value> rejected = pivoted::SelectPivot(Index{0}, assembly, work, options);
  EXPECT_EQ(rejected.status.code, LuCode::ZeroPivot);
  EXPECT_EQ(rejected.status.pivot_column, 0);
  options.pivot_tolerance = 0;
  work.values[0] = std::numeric_limits<Value>::quiet_NaN();
  EXPECT_EQ(pivoted::SelectPivot(Index{0}, assembly, work, options).status.code, LuCode::ZeroPivot);
}

TEST(PivotedFactorize, IndependentRowAndColumnPermutations)
{
  const Matrix A = MakeExtensionInput();
  const symbolic::Analysis<Index> analysis = MakeExtensionAnalysis();
  const pivoted::Factors<Index, Value> factors = pivoted::Factorize(A, analysis);
  ASSERT_EQ(factors.status.code, LuCode::Success);
  EXPECT_EQ(factors.col_order, (std::vector<Index>{2, 3, 0, 1}));
  EXPECT_EQ(factors.row_order, (std::vector<Index>{3, 2, 1, 0}));
  const std::vector<Value> original = Dense(A);
  const std::vector<Value> lower = Dense(factors.lower);
  const std::vector<Value> upper = Dense(factors.upper);
  for (Index j = 0; j < 4; ++j) {
    for (Index i = 0; i < 4; ++i) {
      Value product = 0;
      for (Index k = 0; k < 4; ++k) product += lower[i + k * 4] * upper[k + j * 4];
      EXPECT_NEAR(product, original[factors.row_order[i] + factors.col_order[j] * 4], kTolerance);
    }
  }
}

TEST(PivotedSolve, NormalTransposeAndAdjointWithPaddedMultipleRhs)
{
  const pivoted::Factors<Index, Value> factors = MakeExtensionFactors();
  Value rhs_storage[12];
  Value solution_storage[14];
  DenseMatrix<Index, Value> B(rhs_storage, 4, 2, 6);
  DenseMatrix<Index, Value> X(solution_storage, 4, 2, 7);
  for (pivoted::Operation operation :
       {pivoted::Operation::Normal, pivoted::Operation::Transpose, pivoted::Operation::ConjugateTranspose}) {
    std::fill(std::begin(rhs_storage), std::end(rhs_storage), Value{-99});
    std::fill(std::begin(solution_storage), std::end(solution_storage), Value{-99});
    const Value normal[] = {7, 7, 16, 26};
    const Value transpose[] = {2, 8, 9, 32};
    for (Index j = 0; j < 2; ++j) {
      for (Index i = 0; i < 4; ++i) {
        B[i, j] = (j + 1) * (operation == pivoted::Operation::Normal ? normal[i] : transpose[i]);
      }
    }
    pivoted::Solve(factors, B, X, operation);
    for (Index j = 0; j < 2; ++j) {
      for (Index i = 0; i < 4; ++i) EXPECT_NEAR((X[i, j]), (j + 1) * (i + 1), kTolerance);
      for (Index i = 4; i < 7; ++i) EXPECT_EQ(solution_storage[i + j * 7], -99);
      for (Index i = 4; i < 6; ++i) EXPECT_EQ(rhs_storage[i + j * 6], -99);
    }
  }
}

TEST(ExpertSolve, AppliesScalingOnBothSidesForNormalAndTranspose)
{
  const expert::State<Index, Value> state = MakeExtensionState();
  DenseMatrix<Index, Value> B(4, 1);
  DenseMatrix<Index, Value> X(4, 1);
  const Value normal[] = {7, 7, 16, 26};
  const Value transpose[] = {2, 8, 9, 32};
  for (pivoted::Operation operation : {pivoted::Operation::Normal, pivoted::Operation::Transpose}) {
    for (Index i = 0; i < 4; ++i)
      B[i, 0] = operation == pivoted::Operation::Normal ? normal[i] : transpose[i];
    expert::Solve(state, B, X, operation);
    for (Index i = 0; i < 4; ++i) EXPECT_NEAR((X[i, 0]), i + 1, kTolerance);
  }
}

TEST(ExpertRefinement, CorrectsInitialGuessAndReportsOriginalSystemError)
{
  const Matrix A = MakeExtensionInput();
  const expert::State<Index, Value> state = MakeExtensionState();
  DenseMatrix<Index, Value> B(4, 1);
  DenseMatrix<Index, Value> X(4, 1);
  const Value rhs[] = {7, 7, 16, 26};
  for (Index i = 0; i < 4; ++i) {
    B[i, 0] = rhs[i];
    X[i, 0] = 0;
  }
  const expert::Diagnostics<Value> report = expert::Refine(A, state, B, X);
  ASSERT_EQ(report.backward_error.size(), 1u);
  ASSERT_EQ(report.forward_error_bound.size(), 1u);
  ASSERT_EQ(report.refinement.size(), 1u);
  ASSERT_EQ(report.refinement_steps.size(), 1u);
  EXPECT_EQ(report.refinement[0], expert::RefinementCode::Converged);
  EXPECT_GE(report.refinement_steps[0], 1);
  EXPECT_LE(report.backward_error[0], 1e-12);
  EXPECT_TRUE(std::isfinite(report.forward_error_bound[0]));
  EXPECT_GE(report.forward_error_bound[0], 0);
  EXPECT_GT(report.reciprocal_condition, 0);
  EXPECT_LE(report.reciprocal_condition, 1);
  EXPECT_DOUBLE_EQ(report.reciprocal_pivot_growth, 1);
  for (Index i = 0; i < 4; ++i) EXPECT_NEAR((X[i, 0]), i + 1, kTolerance);
}

TEST(ExpertRefactorize, ChangedStructureDoesNotOverwriteOldState)
{
  Matrix A = MakeExtensionInput();
  expert::State<Index, Value> state = MakeExtensionState();
  A.row_id[0] = 0;
  EXPECT_THROW(expert::Refactorize(A, state), std::invalid_argument);
  EXPECT_EQ(state.factors.status.code, LuCode::Success);
  EXPECT_EQ(state.factors.row_order, (std::vector<Index>{3, 2, 1, 0}));
  EXPECT_DOUBLE_EQ(state.factors.upper.val[0], 10);
}
