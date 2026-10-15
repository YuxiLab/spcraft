#include "SpCraft.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mpi.h>

#include "SummaTestCommon.h"

using namespace spcraft;

constexpr int kLocalExtent = 2;

template <class IT, class NT, class OT, class Matrix>
Matrix ExtractBlock(const Ref<NT>& matrix, GridPtr grid)
{
  const int coordinates[2] = {grid->Row(), grid->Col()};
  const int first_row = coordinates[0] * kLocalExtent;
  const int first_column = coordinates[1] * kLocalExtent;
  OT nonzeros = 0;
  for (int row = 0; row < kLocalExtent; ++row) {
    for (int column = 0; column < kLocalExtent; ++column) {
      if (matrix(first_row + row, first_column + column) != NT{0}) ++nonzeros;
    }
  }

  CooMatrix<IT, NT, OT> coo;
  coo.Allocate(nonzeros, kLocalExtent, kLocalExtent);
  OT position = 0;
  for (int row = 0; row < kLocalExtent; ++row) {
    for (int column = 0; column < kLocalExtent; ++column) {
      const NT value = matrix(first_row + row, first_column + column);
      if (value != NT{0}) {
        coo.entries[position++] = {static_cast<IT>(row), static_cast<IT>(column), value};
      }
    }
  }
  CscMatrix<IT, NT, OT> csc = coo.ToCsc();
  const IT shape_m = static_cast<IT>(matrix.rows());
  const IT shape_n = static_cast<IT>(matrix.cols());
  if constexpr (std::is_same_v<typename Matrix::MatrixType, CscMatrix<IT, NT, OT>>) {
    Matrix block(grid, std::move(csc), shape_m, shape_n);
    return block;
  } else {
    DcscMatrix<IT, NT, OT> local = DcscMatrix<IT, NT, OT>::FromCsc(csc);
    Matrix block(grid, std::move(local), shape_m, shape_n);
    return block;
  }
}

template <class IT, class NT, class OT, class Matrix>
void CheckProduct(const Ref<NT>& a, const Ref<NT>& b, GridPtr grid)
{
  if (grid->Rows() != grid->Cols()) GTEST_SKIP() << "The numerical examples use a square grid";
  const int coordinates[2] = {grid->Row(), grid->Col()};
  SCOPED_TRACE(testing::Message() << "Grid rank " << grid->Rank() << ", block (" << coordinates[0] << ", "
                                  << coordinates[1] << ")");
  const Matrix distributed_a = ExtractBlock<IT, NT, OT, Matrix>(a, grid);
  const Matrix distributed_b = ExtractBlock<IT, NT, OT, Matrix>(b, grid);
  const DistSpCOO<IT, NT, OT> product = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(distributed_a, distributed_b);
  EXPECT_EQ(product.commgrid, grid);
  EXPECT_EQ(product.m, a.rows());
  EXPECT_EQ(product.n, b.cols());
  const CooMatrix<IT, NT, OT>& actual = product.local;

  // Every rank computes an independent global Eigen reference, then compares
  // its own block. A local product alone misses contributions from other ranks.
  const Ref<NT> expected = a * b;
  const int first_row = coordinates[0] * kLocalExtent;
  const int first_column = coordinates[1] * kLocalExtent;
  OT expected_nonzeros = 0;
  for (int row = 0; row < kLocalExtent; ++row) {
    for (int column = 0; column < kLocalExtent; ++column) {
      if (expected(first_row + row, first_column + column) != NT{0}) ++expected_nonzeros;
    }
  }

  EXPECT_EQ(actual.m, kLocalExtent);
  EXPECT_EQ(actual.n, kLocalExtent);
  EXPECT_EQ(actual.nnz, expected_nonzeros) << "Output block has the wrong nonzero count";
  ASSERT_GE(actual.nnz, OT{0});
  if (actual.nnz != OT{0}) ASSERT_NE(actual.entries, nullptr);

  Ref<NT> values = Ref<NT>::Zero(kLocalExtent, kLocalExtent);
  for (OT position = 0; position < actual.nnz; ++position) {
    const TupleEntry<IT, NT>& entry = actual.entries[position];
    ASSERT_GE(entry.row, IT{0});
    ASSERT_LT(entry.row, kLocalExtent);
    ASSERT_GE(entry.col, IT{0});
    ASSERT_LT(entry.col, kLocalExtent);
    values(entry.row, entry.col) += entry.val;
  }

  constexpr NT kTolerance = std::is_same_v<NT, float> ? NT{1e-6} : NT{1e-12};
  for (int row = 0; row < kLocalExtent; ++row) {
    for (int column = 0; column < kLocalExtent; ++column) {
      EXPECT_NEAR(values(row, column), expected(first_row + row, first_column + column), kTolerance)
          << "Wrong value at local (" << row << ", " << column << ")";
    }
  }
}

template <class Types>
class Summa2DTest : public testing::Test
{
 protected:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  std::shared_ptr<const CommGrid2D> grid{};  //!< Shared square grid for each case.
  int width = 0;

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  void SetUp() override
  {
    grid = std::make_shared<CommGrid2D>();
    width = grid->Rows();
  }

  void TearDown() override { grid.reset(); }
};

TYPED_TEST_SUITE(Summa2DTest, DistributedSummaTypes);

TYPED_TEST(Summa2DTest, NonemptyProductMatchesEigen)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const int extent = kLocalExtent * this->width;
  Ref<NT> a = Ref<NT>::Zero(extent, extent);
  Ref<NT> b = Ref<NT>::Zero(extent, extent);
  // A = 2I + cyclic forward shift; B = 3I + cyclic backward shift.
  // On a 2x2 process grid their product includes contributions across blocks.
  for (int row = 0; row < extent; ++row) {
    a(row, row) = NT{2};
    a(row, (row + 1) % extent) = NT{1};
    b(row, row) = NT{3};
    b(row, (row + extent - 1) % extent) = NT{1};
  }
  CheckProduct<IT, NT, OT, Matrix>(a, b, this->grid);
}

TYPED_TEST(Summa2DTest, RemoteOperandsReachAnInitiallyEmptyBlock)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const int extent = kLocalExtent * this->width;
  Ref<NT> a = Ref<NT>::Zero(extent, extent);
  Ref<NT> b = Ref<NT>::Zero(extent, extent);
  a(0, extent - 1) = NT{2};
  b(extent - 1, 0) = NT{3};
  // With four ranks, block (0, 0) starts with empty local operands but must
  // produce C(0, 0) = 6 after receiving the remote panels.
  CheckProduct<IT, NT, OT, Matrix>(a, b, this->grid);
}

TYPED_TEST(Summa2DTest, EmptyProductKeepsLocalShape)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const int extent = kLocalExtent * this->width;
  const Ref<NT> empty = Ref<NT>::Zero(extent, extent);
  CheckProduct<IT, NT, OT, Matrix>(empty, empty, this->grid);
}

TYPED_TEST(Summa2DTest, RejectsIncompatibleInnerDimensions)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const IT extent = static_cast<IT>(kLocalExtent * this->width);
  const IT a_shape_m = extent;
  const IT a_shape_n = extent + 1;
  const IT b_shape_m = extent;
  const IT b_shape_n = extent;
  const Matrix a(this->grid, a_shape_m, a_shape_n);
  const Matrix b(this->grid, b_shape_m, b_shape_n);
  using Ring = PlusTimesRing<NT>;
  EXPECT_THROW(static_cast<void>(Mpi2DSummaSpGEMM<Ring>(a, b)), std::invalid_argument);
}

TYPED_TEST(Summa2DTest, EmptyProductRetainsBlockCyclicShape)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const IT a_shape_m = 11;
  const IT a_shape_n = 7;
  const IT b_shape_m = 7;
  const IT b_shape_n = 13;
  const Matrix a(this->grid, a_shape_m, a_shape_n);
  const Matrix b(this->grid, b_shape_m, b_shape_n);
  const DistSpCOO<IT, NT, OT> result = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(a, b);
  EXPECT_EQ(result.commgrid, this->grid);
  EXPECT_EQ(result.m, a_shape_m);
  EXPECT_EQ(result.n, b_shape_n);
  EXPECT_EQ(result.local.m, a.LocalRows());
  EXPECT_EQ(result.local.n, b.LocalCols());
  EXPECT_EQ(result.local.nnz, 0);
  EXPECT_EQ(result.GlobalNnz(), 0);
}

TYPED_TEST(Summa2DTest, ZeroInnerDimensionKeepsOuterDimensions)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const IT a_shape_m = 11;
  const IT a_shape_n = 0;
  const IT b_shape_m = 0;
  const IT b_shape_n = 13;
  const Matrix a(this->grid, a_shape_m, a_shape_n);
  const Matrix b(this->grid, b_shape_m, b_shape_n);
  const DistSpCOO<IT, NT, OT> result = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(a, b);
  EXPECT_EQ(result.m, 11);
  EXPECT_EQ(result.n, 13);
  EXPECT_EQ(result.local.m, a.LocalRows());
  EXPECT_EQ(result.local.n, b.LocalCols());
  EXPECT_EQ(result.local.nnz, 0);
}

TYPED_TEST(Summa2DTest, RejectsMissingOrDifferentGrids)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const IT shape_m = 4;
  const IT shape_n = 4;
  const Matrix empty;
  const Matrix a(this->grid, shape_m, shape_n);
  using Ring = PlusTimesRing<NT>;
  EXPECT_THROW(static_cast<void>(Mpi2DSummaSpGEMM<Ring>(empty, empty)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Mpi2DSummaSpGEMM<Ring>(a, empty)), std::invalid_argument);
  std::shared_ptr<const CommGrid2D> other_grid = std::make_shared<CommGrid2D>();
  const Matrix b(other_grid, shape_m, shape_n);
  EXPECT_THROW(static_cast<void>(Mpi2DSummaSpGEMM<Ring>(a, b)), std::invalid_argument);
}

TYPED_TEST(Summa2DTest, ResultKeepsGridAliveAfterInputsAreDestroyed)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const IT shape_m = 4;
  const IT shape_n = 4;
  std::weak_ptr<const CommGrid2D> lifetime = this->grid;
  DistSpCOO<IT, NT, OT> result;
  {
    const Matrix a(this->grid, shape_m, shape_n);
    const Matrix b(this->grid, shape_m, shape_n);
    result = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(a, b);
  }
  this->grid.reset();
  EXPECT_FALSE(lifetime.expired());
  int size = 0;
  MPI_Comm_size(result.commgrid->World(), &size);
  EXPECT_EQ(size, result.commgrid->Size());
}

TYPED_TEST(Summa2DTest, ContiguousSubmatricesAndEdgesMatchEigen)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  // Reversing ranks also checks that roots use coordinates, rather than world ranks.
  const int side = this->grid->Rows();
  const auto reversed = [](int rank, int rows, int cols) {
    const int mapped = rows * cols - rank - 1;
    return std::pair<int, int>{mapped / cols, mapped % cols};
  };
  const std::shared_ptr<const CommGrid2D> grid =
      std::make_shared<CommGrid2D>(MPI_COMM_WORLD, side, side, reversed);
  const IT a_shape_m = 5;
  const IT a_shape_n = 11;
  const IT b_shape_m = 11;
  const IT b_shape_n = 7;
  Ref<NT> a = Ref<NT>::Zero(5, 11);
  Ref<NT> b = Ref<NT>::Zero(11, 7);
  for (IT i = 0; i < 5; ++i)
    for (IT k = 0; k < 11; ++k)
      if ((i + k) % 3 != 0) a(i, k) = static_cast<NT>((i + k) % 5 - 2);
  for (IT k = 0; k < 11; ++k)
    for (IT j = 0; j < 7; ++j)
      if ((k + j) % 2 == 0) b(k, j) = static_cast<NT>((k + j) % 4 + 1);
  const auto owns_a = [](IT, IT) { return true; };
  const auto owns_b = [](IT, IT) { return true; };
  const Matrix da = MakeSummaInput<IT, NT, OT, Matrix>(a, grid, a_shape_m, a_shape_n, owns_a);
  const Matrix db = MakeSummaInput<IT, NT, OT, Matrix>(b, grid, b_shape_m, b_shape_n, owns_b);
  const Ref<NT> expected = a * b;
  for (const int threads : {1, 2}) {
    OMP_SET_NUM_THREADS(threads);
    const DistSpCOO<IT, NT, OT> result = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(da, db);
    CheckSummaValues(result, expected);
  }
  OMP_SET_NUM_THREADS(1);
}

TYPED_TEST(Summa2DTest, StageCountsVaryAcrossOwnersAndEdgeBlocks)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const IT a_shape_m = 3;
  const IT a_shape_n = 13;
  const IT b_shape_m = 13;
  const IT b_shape_n = 4;
  Ref<NT> a = Ref<NT>::Zero(3, 13);
  Ref<NT> b = Ref<NT>::Zero(13, 4);
  // Empty, dense, sparse, empty, and short edge stages require different buffer sizes.
  for (IT k = 3; k < 6; ++k) {
    for (IT i = 0; i < 3; ++i) a(i, k) = static_cast<NT>(i + k);
    for (IT j = 0; j < 4; ++j) b(k, j) = static_cast<NT>(k - j);
  }
  a(0, 6) = NT{2};
  b(6, 0) = NT{3};
  for (IT i = 0; i < 3; ++i) a(i, 12) = static_cast<NT>(i + 1);
  for (IT j = 0; j < 4; ++j) b(12, j) = static_cast<NT>(j + 1);
  const CommGrid2D& grid = *this->grid;
  const auto owns_a = [](IT, IT) { return true; };
  const auto owns_b = [](IT, IT) { return true; };
  const Matrix da = MakeSummaInput<IT, NT, OT, Matrix>(a, this->grid, a_shape_m, a_shape_n, owns_a);
  const Matrix db = MakeSummaInput<IT, NT, OT, Matrix>(b, this->grid, b_shape_m, b_shape_n, owns_b);
  const DistSpCOO<IT, NT, OT> result = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(da, db);
  const Ref<NT> expected = a * b;
  CheckSummaValues(result, expected);
}

TYPED_TEST(Summa2DTest, CancellationRetainsOneStructuralEntry)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const IT a_shape_m = 1;
  const IT a_shape_n = 3;
  const IT b_shape_m = 3;
  const IT b_shape_n = 1;
  Ref<NT> a(1, 3);
  a << NT{2}, NT{0}, NT{-2};
  const Ref<NT> b = Ref<NT>::Ones(3, 1);
  const CommGrid2D& grid = *this->grid;
  const auto owns = [](IT, IT) { return true; };
  const Matrix da = MakeSummaInput<IT, NT, OT, Matrix>(a, this->grid, a_shape_m, a_shape_n, owns);
  const Matrix db = MakeSummaInput<IT, NT, OT, Matrix>(b, this->grid, b_shape_m, b_shape_n, owns);
  const DistSpCOO<IT, NT, OT> result = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(da, db);
  EXPECT_EQ(result.GlobalNnz(), 1);
  const Ref<NT> expected = a * b;
  CheckSummaValues(result, expected);
}

TYPED_TEST(Summa2DTest, DuplicateInputEntriesAreCombined)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const IT shape_m = 1;
  const IT shape_n = 1;
  Matrix a(this->grid, shape_m, shape_n);
  CooMatrix<IT, NT, OT> coo;
  const bool owns = a.Owns(0, 0);
  coo.Allocate(owns ? OT{2} : OT{0}, a.LocalRows(), a.LocalCols());
  if (owns) {
    coo.entries[0] = {0, 0, NT{1}};
    coo.entries[1] = {0, 0, NT{2}};
  }
  CscMatrix<IT, NT, OT> csc = coo.ToCsc();
  if constexpr (std::is_same_v<typename Matrix::MatrixType, CscMatrix<IT, NT, OT>>) {
    a.local = std::move(csc);
  } else {
    a.local = DcscMatrix<IT, NT, OT>::FromCsc(csc);
  }
  const DistSpCOO<IT, NT, OT> result = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(a, a);
  const Ref<NT> expected = Ref<NT>::Constant(1, 1, NT{9});
  EXPECT_EQ(result.GlobalNnz(), 1);
  CheckSummaValues(result, expected);
}

TYPED_TEST(Summa2DTest, ZeroOuterDimensions)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  for (const IT outer : {IT{0}, IT{3}}) {
    const IT a_shape_m = outer;
    const IT a_shape_n = 5;
    const IT b_shape_m = 5;
    const IT b_shape_n = 3 - outer;
    const Matrix a(this->grid, a_shape_m, a_shape_n);
    const Matrix b(this->grid, b_shape_m, b_shape_n);
    const DistSpCOO<IT, NT, OT> result = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(a, b);
    const Ref<NT> expected = Ref<NT>::Zero(outer, 3 - outer);
    CheckSummaValues(result, expected);
    EXPECT_EQ(result.GlobalNnz(), 0);
  }
}

TYPED_TEST(Summa2DTest, RejectsRectangularGrid)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  if (this->grid->Size() == 1) GTEST_SKIP() << "Needs more than one process";
  const std::shared_ptr<const CommGrid2D> grid =
      std::make_shared<CommGrid2D>(MPI_COMM_WORLD, 1, this->grid->Size());
  const IT shape_m = 4;
  const IT shape_n = 4;
  const Matrix a(grid, shape_m, shape_n);
  using Ring = PlusTimesRing<NT>;
  EXPECT_THROW((void)Mpi2DSummaSpGEMM<Ring>(a, a), std::invalid_argument);
}

int TestRun(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

int main(int argc, char** argv)
{
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    MPI_Finalize();
    return 1;
  }
  const int status = TestRun(argc, argv);
  int global_status = 0;
  MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return global_status;
}
