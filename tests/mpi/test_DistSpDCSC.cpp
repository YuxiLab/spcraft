#include "DistMatrixTest.h"

using namespace spcraft;

template <class Types>
using DistSpDCSCTest = DistributedMatrixTest<DistSpDCSC, Types>;

TYPED_TEST_SUITE(DistSpDCSCTest, DistributedTypes);

TYPED_TEST(DistSpDCSCTest, DefaultAndEmpty)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckDefaultAndEmpty<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpDCSCTest, CloneAndSharedGridOwnership)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckCloneAndSharedOwnership<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpDCSCTest, MoveConstructionAssignmentAndSelfMove)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckMoves<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpDCSCTest, ViewsAndReset)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;
  CheckViewAndReset<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpDCSCTest, ContiguousIndicesMatchGlobalEnumeration)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckContiguousIndices<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpDCSCTest, EmptyRanksAndLargeIndices)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckEmptyRanksAndIntegerLimits<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpDCSCTest, InvalidInputAndStrongAllocationGuarantee)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckInvalidInputAndFailedAllocation<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpDCSCTest, StoredColumnCountsAreExplicitAndPreservedOnFailure)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  using Matrix = typename TestFixture::Matrix;
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const IT shape_m = 2 * grid->Rows();
  const IT shape_n = 2 * grid->Cols();
  Matrix matrix(grid, shape_m, shape_n);
  matrix.Allocate(3, 2);
  EXPECT_EQ(matrix.local.nnz, 3);
  EXPECT_EQ(matrix.local.nzc, 2);
  const void* buffer = matrix.local.val;
  EXPECT_THROW(matrix.Allocate(3, 0), std::invalid_argument);
  EXPECT_THROW(matrix.Allocate(3, 3), std::invalid_argument);
  EXPECT_THROW(matrix.Allocate(0, 1), std::invalid_argument);
  EXPECT_EQ(matrix.local.val, buffer);
  EXPECT_EQ(matrix.local.nzc, 2);
  matrix.Allocate(0, 0);
  EXPECT_EQ(matrix.local.nzc, 0);
  EXPECT_EQ(matrix.local.nnz, 0);
}

TYPED_TEST(DistSpDCSCTest, ClonePreservesCompressedLogicalColumnIds)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  constexpr IT kLocalRows = 2;
  constexpr IT kLocalCols = 4;
  const IT shape_m = kLocalRows * grid->Rows();
  const IT shape_n = kLocalCols * grid->Cols();
  CooMatrix<IT, NT, OT> coo;
  coo.Allocate(1, kLocalRows, kLocalCols);
  coo.entries[0] = {0, kLocalCols - 1, static_cast<NT>(5)};
  const CscMatrix<IT, NT, OT> csc = coo.ToCsc();
  DcscMatrix<IT, NT, OT> local = DcscMatrix<IT, NT, OT>::FromCsc(csc);
  const Matrix matrix(grid, std::move(local), shape_m, shape_n);
  const Matrix cloned = matrix.Clone();
  EXPECT_EQ(cloned.local.n, kLocalCols);
  ASSERT_EQ(cloned.local.nzc, 1);
  EXPECT_EQ(cloned.local.col_id[0], kLocalCols - 1);
  EXPECT_EQ(cloned.local.col_ptr[0], 0);
  EXPECT_EQ(cloned.local.col_ptr[1], 1);
  EXPECT_EQ(cloned.local.row_id[0], 0);
  EXPECT_EQ(cloned.local.val[0], static_cast<NT>(5));
  EXPECT_NE(cloned.local.col_id, matrix.local.col_id);
  EXPECT_NE(cloned.local.col_ptr, matrix.local.col_ptr);
  EXPECT_NE(cloned.local.row_id, matrix.local.row_id);
  EXPECT_NE(cloned.local.val, matrix.local.val);
  EXPECT_EQ(cloned.GlobalCol(cloned.local.col_id[0]), kLocalCols * grid->Col() + kLocalCols - 1);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  const int status = TestRun(argc, argv);
  int global_status = 0;
  MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return global_status;
}
