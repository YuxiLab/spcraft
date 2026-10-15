#include "SpCraft.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>

using namespace spcraft;

TEST(CommGrid3DTest, CoordinatesOwnershipAndCommunicators)
{
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int layers = size == 8 ? 2 : size;
  const CommGrid3D grid(MPI_COMM_WORLD, layers);
  const CommGrid2D& plane = *grid.Plane();
  EXPECT_EQ(plane.Rows(), plane.Cols());
  EXPECT_EQ(grid.Layer(), rank / plane.Size());
  EXPECT_EQ(grid.Layers(), layers);
  EXPECT_EQ(grid.RankAt(plane.Row(), plane.Col(), grid.Layer()), rank);

  // All three communicator axes use coordinates as ranks, including fibers.
  int fiber_rank = -1;
  int fiber_size = 0;
  MPI_Comm_rank(grid.FiberWorld(), &fiber_rank);
  MPI_Comm_size(grid.FiberWorld(), &fiber_size);
  EXPECT_EQ(fiber_rank, grid.Layer());
  EXPECT_EQ(fiber_size, layers);
  int value = grid.Layer() == layers - 1 ? plane.Rank() : -1;
  MPI_Bcast(&value, 1, MPI_INT, layers - 1, grid.FiberWorld());
  EXPECT_EQ(value, plane.Rank());
  for (int64_t i = 0; i < 7; ++i) {
    for (int64_t k = 0; k < 11; ++k) {
      const int expected_a =
          (k % layers) * plane.Size() + (i % plane.Rows()) * plane.Cols() + k % plane.Cols();
      const int expected_b =
          (i % layers) * plane.Size() + (i % plane.Rows()) * plane.Cols() + k % plane.Cols();
      EXPECT_EQ(grid.AOwner(i, k), expected_a);
      EXPECT_EQ(grid.BOwner(i, k), expected_b);
    }
  }
  EXPECT_THROW(grid.RankAt(-1, 0, 0), std::out_of_range);
  EXPECT_THROW(grid.RankAt(0, 0, layers), std::out_of_range);
  EXPECT_THROW(grid.AOwner(-1, 0), std::out_of_range);
  EXPECT_THROW(grid.BOwner(0, -1), std::out_of_range);
}

TEST(CommGrid3DTest, MovesAndSharedPlaneLifetime)
{
  static_assert(!std::is_copy_constructible_v<CommGrid3D>);
  static_assert(std::is_nothrow_move_constructible_v<CommGrid3D>);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int layers = size == 8 ? 2 : size;
  std::shared_ptr<const CommGrid2D> retained;
  {
    CommGrid3D source(MPI_COMM_WORLD, layers);
    retained = source.Plane();
    const MPI_Comm fiber = source.FiberWorld();
    CommGrid3D moved(std::move(source));
    EXPECT_EQ(source.World(), MPI_COMM_NULL);
    EXPECT_EQ(source.FiberWorld(), MPI_COMM_NULL);
    EXPECT_FALSE(source.Plane());
    EXPECT_EQ(moved.FiberWorld(), fiber);
    CommGrid3D assigned(MPI_COMM_WORLD, layers);
    assigned = std::move(moved);
    EXPECT_FALSE(moved.Plane());
    EXPECT_EQ(assigned.Plane(), retained);
    CommGrid3D& self = assigned;
    assigned = std::move(self);
    EXPECT_EQ(assigned.FiberWorld(), fiber);
  }
  int plane_size = 0;
  MPI_Comm_size(retained->World(), &plane_size);
  EXPECT_EQ(plane_size, size / layers);
}

TEST(CommGrid3DTest, RejectsInvalidOrInconsistentDimensions)
{
  EXPECT_THROW(CommGrid3D(MPI_COMM_NULL, 1), std::invalid_argument);
  EXPECT_THROW(CommGrid3D(MPI_COMM_WORLD, 0), std::invalid_argument);
  EXPECT_THROW(CommGrid3D(MPI_COMM_WORLD, -1), std::invalid_argument);
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  EXPECT_THROW(CommGrid3D(MPI_COMM_WORLD, size + 1), std::invalid_argument);
  if (size > 1) {
    const int layers = rank == 0 ? 1 : size;
    EXPECT_THROW(CommGrid3D(MPI_COMM_WORLD, layers), std::invalid_argument);
    EXPECT_THROW(CommGrid3D(MPI_COMM_WORLD, size / 2), std::invalid_argument);
  }
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
