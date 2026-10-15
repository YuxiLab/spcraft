#include "mpi/CommGrid3D.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace spcraft
{

CommGrid3D::CommGrid3D(MPI_Comm world, int layers)
{
  if (world == MPI_COMM_NULL) throw std::invalid_argument("CommGrid3D needs a communicator");
  int intercommunicator = 0;
  MPI_Comm_test_inter(world, &intercommunicator);
  if (intercommunicator) throw std::invalid_argument("CommGrid3D needs an intracommunicator");
  int minimum = 0;
  int maximum = 0;
  MPI_Allreduce(&layers, &minimum, 1, MPI_INT, MPI_MIN, world);
  MPI_Allreduce(&layers, &maximum, 1, MPI_INT, MPI_MAX, world);
  if (minimum != maximum || layers <= 0) {
    throw std::invalid_argument("CommGrid3D needs the same positive layer count on every rank");
  }
  int size = 0;
  int rank = 0;
  MPI_Comm_size(world, &size);
  MPI_Comm_rank(world, &rank);
  const int plane_size = size / layers;
  const int side = static_cast<int>(std::sqrt(plane_size));
  if (size % layers != 0 || side == 0 || static_cast<int64_t>(side) * side != plane_size) {
    throw std::invalid_argument("CommGrid3D needs layers times a square number of ranks");
  }
  layers_ = layers;
  layer_ = rank / plane_size;

  // Plane ranks retain row-major order; fiber ranks are ordered by layer.
  MPI_Comm_dup(world, &world_);
  MPI_Comm plane = MPI_COMM_NULL;
  MPI_Comm_split(world_, layer_, rank % plane_size, &plane);
  try {
    plane_ = std::make_shared<CommGrid2D>(plane, side, side);
  } catch (...) {
    MPI_Comm_free(&plane);
    Free();
    throw;
  }
  MPI_Comm_free(&plane);
  MPI_Comm_split(world_, rank % plane_size, layer_, &fiber_);
}

CommGrid3D::CommGrid3D(CommGrid3D&& rhs) noexcept { *this = std::move(rhs); }

CommGrid3D& CommGrid3D::operator=(CommGrid3D&& rhs) noexcept
{
  if (this != &rhs) {
    Free();
    world_ = std::exchange(rhs.world_, MPI_COMM_NULL);
    fiber_ = std::exchange(rhs.fiber_, MPI_COMM_NULL);
    plane_ = std::move(rhs.plane_);
    layers_ = std::exchange(rhs.layers_, 0);
    layer_ = std::exchange(rhs.layer_, -1);
  }
  return *this;
}

CommGrid3D::~CommGrid3D() { Free(); }

MPI_Comm CommGrid3D::World() const noexcept { return world_; }

MPI_Comm CommGrid3D::FiberWorld() const noexcept { return fiber_; }

const CommGrid3D::PlanePtr& CommGrid3D::Plane() const noexcept { return plane_; }

int CommGrid3D::Layers() const noexcept { return layers_; }

int CommGrid3D::Layer() const noexcept { return layer_; }

int CommGrid3D::RankAt(int row, int col, int layer) const
{
  if (!plane_ || layer < 0 || layer >= layers_) {
    throw std::out_of_range("CommGrid3D layer is outside the grid");
  }
  return layer * plane_->Size() + plane_->RankAt(row, col);
}

int CommGrid3D::AOwner(int64_t block_row, int64_t inner_block) const
{
  if (!plane_ || block_row < 0 || inner_block < 0) {
    throw std::out_of_range("CommGrid3D block coordinates must be nonnegative on a live grid");
  }
  const int row = static_cast<int>(block_row % plane_->Rows());
  const int col = static_cast<int>(inner_block % plane_->Cols());
  const int layer = static_cast<int>(inner_block % layers_);
  return RankAt(row, col, layer);
}

int CommGrid3D::BOwner(int64_t inner_block, int64_t block_col) const
{
  if (!plane_ || inner_block < 0 || block_col < 0) {
    throw std::out_of_range("CommGrid3D block coordinates must be nonnegative on a live grid");
  }
  const int row = static_cast<int>(inner_block % plane_->Rows());
  const int col = static_cast<int>(block_col % plane_->Cols());
  const int layer = static_cast<int>(inner_block % layers_);
  return RankAt(row, col, layer);
}

void CommGrid3D::Free() noexcept
{
  plane_.reset();
  if (fiber_ != MPI_COMM_NULL) MPI_Comm_free(&fiber_);
  if (world_ != MPI_COMM_NULL) MPI_Comm_free(&world_);
}

}  // namespace spcraft
