#include "SpCraft.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using namespace spcraft;

static_assert(!std::is_copy_constructible_v<CommGrid2D>);
static_assert(!std::is_copy_assignable_v<CommGrid2D>);
static_assert(std::is_nothrow_move_constructible_v<CommGrid2D>);
static_assert(std::is_nothrow_move_assignable_v<CommGrid2D>);

void CheckCommunicators(const CommGrid2D& grid)
{
  int rank = -1;
  int size = 0;
  int comparison = MPI_UNEQUAL;
  MPI_Comm_compare(MPI_COMM_WORLD, grid.World(), &comparison);
  EXPECT_EQ(comparison, MPI_CONGRUENT) << "The grid must own a duplicate of the source";
  MPI_Comm_rank(grid.World(), &rank);
  MPI_Comm_size(grid.World(), &size);
  EXPECT_EQ(rank, grid.Rank());
  EXPECT_EQ(size, grid.Size());
  EXPECT_EQ(grid.RankAt(grid.Row(), grid.Col()), rank);

  MPI_Comm_rank(grid.RowWorld(), &rank);
  MPI_Comm_size(grid.RowWorld(), &size);
  EXPECT_EQ(rank, grid.Col());
  EXPECT_EQ(size, grid.Cols());
  std::vector<int> row_ranks(grid.Cols());
  const int world_rank = grid.Rank();
  MPI_Allgather(&world_rank, 1, MPI_INT, row_ranks.data(), 1, MPI_INT, grid.RowWorld());
  for (int col = 0; col < grid.Cols(); ++col) {
    EXPECT_EQ(row_ranks[col], grid.RankAt(grid.Row(), col)) << "Row communicator column " << col;
  }

  MPI_Comm_rank(grid.ColWorld(), &rank);
  MPI_Comm_size(grid.ColWorld(), &size);
  EXPECT_EQ(rank, grid.Row());
  EXPECT_EQ(size, grid.Rows());
  std::vector<int> col_ranks(grid.Rows());
  MPI_Allgather(&world_rank, 1, MPI_INT, col_ranks.data(), 1, MPI_INT, grid.ColWorld());
  for (int row = 0; row < grid.Rows(); ++row) {
    EXPECT_EQ(col_ranks[row], grid.RankAt(row, grid.Col())) << "Column communicator row " << row;
  }
}

TEST(CommGrid2D, DefaultShapeIsBalancedAndSquareWhenPossible)
{
  const CommGrid2D grid;
  EXPECT_EQ(grid.Rows() * grid.Cols(), grid.Size());
  EXPECT_GE(grid.Rows(), grid.Cols());
  if (grid.Size() == 1) {
    EXPECT_EQ(grid.Rows(), 1);
    EXPECT_EQ(grid.Cols(), 1);
  } else if (grid.Size() == 4) {
    EXPECT_EQ(grid.Rows(), 2);
    EXPECT_EQ(grid.Cols(), 2);
  } else if (grid.Size() == 6) {
    EXPECT_EQ(grid.Rows(), 3);
    EXPECT_EQ(grid.Cols(), 2);
  }
  EXPECT_EQ(grid.Row(), grid.Rank() / grid.Cols());
  EXPECT_EQ(grid.Col(), grid.Rank() % grid.Cols());
  CheckCommunicators(grid);
}

TEST(CommGrid2D, ExplicitRectanglesAndPartiallySpecifiedDimensions)
{
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const CommGrid2D row(MPI_COMM_WORLD, 1, size);
  EXPECT_EQ(row.Rows(), 1);
  EXPECT_EQ(row.Cols(), size);
  CheckCommunicators(row);
  const CommGrid2D column(MPI_COMM_WORLD, size, 0);
  EXPECT_EQ(column.Rows(), size);
  EXPECT_EQ(column.Cols(), 1);
  CheckCommunicators(column);
  const CommGrid2D constrained(MPI_COMM_WORLD, 0, size);
  EXPECT_EQ(constrained.Rows(), 1);
  EXPECT_EQ(constrained.Cols(), size);
  CheckCommunicators(constrained);
}

TEST(CommGrid2D, CustomColumnMajorLayoutKeepsCommunicatorRanksInCoordinateOrder)
{
  const CommGrid2D::RankLayout layout = [](int rank, int rows, int) {
    return std::pair<int, int>{rank % rows, rank / rows};
  };
  const CommGrid2D grid(MPI_COMM_WORLD, 0, 0, layout);
  EXPECT_EQ(grid.Row(), grid.Rank() % grid.Rows());
  EXPECT_EQ(grid.Col(), grid.Rank() / grid.Rows());
  for (int row = 0; row < grid.Rows(); ++row) {
    for (int col = 0; col < grid.Cols(); ++col) {
      EXPECT_EQ(grid.RankAt(row, col), col * grid.Rows() + row);
    }
  }
  CheckCommunicators(grid);

  // A row root is a column coordinate; a column root is a row coordinate.
  int row_value = grid.Rank();
  const int row_root = grid.Cols() - 1;
  MPI_Bcast(&row_value, 1, MPI_INT, row_root, grid.RowWorld());
  EXPECT_EQ(row_value, row_root * grid.Rows() + grid.Row());
  int col_value = grid.Rank();
  const int col_root = grid.Rows() - 1;
  MPI_Bcast(&col_value, 1, MPI_INT, col_root, grid.ColWorld());
  EXPECT_EQ(col_value, grid.Col() * grid.Rows() + col_root);
}

TEST(CommGrid2D, CustomRectangularLayoutMapsBlocksCyclically)
{
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int rows = size % 2 == 0 ? 2 : 1;
  const int cols = size / rows;
  const CommGrid2D::RankLayout reverse = [](int rank, int grid_rows, int grid_cols) {
    const int reversed_rank = grid_rows * grid_cols - 1 - rank;
    return std::pair<int, int>{reversed_rank / grid_cols, reversed_rank % grid_cols};
  };
  const CommGrid2D grid(MPI_COMM_WORLD, rows, cols, reverse);
  for (int64_t row = 0; row < 3 * rows; ++row) {
    for (int64_t col = 0; col < 3 * cols; ++col) {
      const int expected = size - 1 - static_cast<int>((row % rows) * cols + col % cols);
      EXPECT_EQ(grid.Owner(row, col), expected) << "Block (" << row << ", " << col << ")";
    }
  }
  const int64_t largest = std::numeric_limits<int64_t>::max();
  const int expected = size - 1 - static_cast<int>((largest % rows) * cols + largest % cols);
  EXPECT_EQ(grid.Owner(largest, largest), expected);
  CheckCommunicators(grid);
}

TEST(CommGrid2D, OwnedCommunicatorsSurviveFreeingTheSource)
{
  MPI_Comm source = MPI_COMM_NULL;
  MPI_Comm_dup(MPI_COMM_WORLD, &source);
  const CommGrid2D grid(source);
  MPI_Comm_free(&source);
  CheckCommunicators(grid);
}

TEST(CommGrid2D, UsesRanksRelativeToASubcommunicator)
{
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm source = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, world_rank, &source);
  {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(source, &rank);
    MPI_Comm_size(source, &size);
    const CommGrid2D grid(source, 1, size);
    EXPECT_EQ(grid.Rank(), rank);
    EXPECT_EQ(grid.Owner(0, rank), rank);
    int row_rank = -1;
    MPI_Comm_rank(grid.RowWorld(), &row_rank);
    EXPECT_EQ(row_rank, rank);
  }
  MPI_Comm_free(&source);
}

TEST(CommGrid2D, MovesTransferOwnershipAndSelfMoveKeepsTheGrid)
{
  CommGrid2D source;
  const MPI_Comm world = source.World();
  const MPI_Comm rowworld = source.RowWorld();
  const MPI_Comm colworld = source.ColWorld();
  CommGrid2D moved(std::move(source));
  EXPECT_EQ(source.World(), MPI_COMM_NULL);
  EXPECT_EQ(source.RowWorld(), MPI_COMM_NULL);
  EXPECT_EQ(source.ColWorld(), MPI_COMM_NULL);
  EXPECT_EQ(source.Size(), 0);
  EXPECT_THROW(source.Owner(0, 0), std::logic_error);
  EXPECT_EQ(moved.World(), world);
  EXPECT_EQ(moved.RowWorld(), rowworld);
  EXPECT_EQ(moved.ColWorld(), colworld);

  CommGrid2D destination;
  destination = std::move(moved);
  EXPECT_EQ(moved.World(), MPI_COMM_NULL);
  EXPECT_EQ(moved.RowWorld(), MPI_COMM_NULL);
  EXPECT_EQ(moved.ColWorld(), MPI_COMM_NULL);
  EXPECT_EQ(destination.World(), world);
  CommGrid2D& same = destination;
  destination = std::move(same);
  EXPECT_EQ(destination.World(), world);
  CheckCommunicators(destination);
}

TEST(CommGrid2D, RejectsInvalidShapesAndCoordinates)
{
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  EXPECT_THROW(CommGrid2D(MPI_COMM_NULL), std::invalid_argument);
  EXPECT_THROW(CommGrid2D(MPI_COMM_WORLD, -1, 0), std::invalid_argument);
  EXPECT_THROW(CommGrid2D(MPI_COMM_WORLD, size + 1, 1), std::invalid_argument);
  EXPECT_THROW(CommGrid2D(MPI_COMM_WORLD, size + 1, 0), std::invalid_argument);
  const CommGrid2D grid;
  EXPECT_THROW(grid.RankAt(-1, 0), std::out_of_range);
  EXPECT_THROW(grid.RankAt(0, grid.Cols()), std::out_of_range);
  EXPECT_THROW(grid.RankAt(grid.Rows(), 0), std::out_of_range);
  EXPECT_THROW(grid.Owner(-1, 0), std::out_of_range);
  EXPECT_THROW(grid.Owner(0, -1), std::out_of_range);
}

TEST(CommGrid2D, RejectsInconsistentShapeCollectively)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1) GTEST_SKIP() << "Requires multiple ranks";
  const int rows = rank == 0 ? 1 : 0;
  EXPECT_THROW(CommGrid2D(MPI_COMM_WORLD, rows, 0), std::invalid_argument);
}

TEST(CommGrid2D, RejectsDuplicateAndOutOfRangeMappingsCollectively)
{
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const CommGrid2D::RankLayout outside = [](int rank, int rows, int cols) {
    if (rank == 0) return std::pair<int, int>{rows, 0};
    return std::pair<int, int>{rank / cols, rank % cols};
  };
  EXPECT_THROW(CommGrid2D(MPI_COMM_WORLD, 0, 0, outside), std::invalid_argument);
  if (size == 1) GTEST_SKIP() << "Duplicate ownership requires multiple ranks";
  const CommGrid2D::RankLayout duplicate = [](int, int, int) { return std::pair<int, int>{0, 0}; };
  EXPECT_THROW(CommGrid2D(MPI_COMM_WORLD, 0, 0, duplicate), std::invalid_argument);
}

TEST(CommGrid2D, CallbackFailureOnOneRankIsReportedOnAllRanks)
{
  const CommGrid2D::RankLayout failing = [](int rank, int, int cols) {
    if (rank == 0) throw std::runtime_error("Test callback failure");
    return std::pair<int, int>{rank / cols, rank % cols};
  };
  EXPECT_THROW(CommGrid2D(MPI_COMM_WORLD, 0, 0, failing), std::invalid_argument);
}

int TestRun(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
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
