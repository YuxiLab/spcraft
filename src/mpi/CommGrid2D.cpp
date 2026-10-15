#include "mpi/CommGrid2D.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace spcraft
{

CommGrid2D::CommGrid2D(MPI_Comm world, int rows, int cols) : CommGrid2D(world, rows, cols, RankLayout{}) {}

CommGrid2D::CommGrid2D(MPI_Comm world, int rows, int cols, RankLayout layout)
{
  if (world == MPI_COMM_NULL) throw std::invalid_argument("CommGrid2D needs a communicator");
  int intercommunicator = 0;
  MPI_Comm_test_inter(world, &intercommunicator);
  if (intercommunicator) throw std::invalid_argument("CommGrid2D needs an intracommunicator");
  MPI_Comm_rank(world, &rank_);
  MPI_Comm_size(world, &size_);

  // Agree on shape before any split: divergent requests would build different groups.
  int dimensions[2] = {rows, cols};
  int minimum[2] = {};
  int maximum[2] = {};
  MPI_Allreduce(dimensions, minimum, 2, MPI_INT, MPI_MIN, world);
  MPI_Allreduce(dimensions, maximum, 2, MPI_INT, MPI_MAX, world);
  if (minimum[0] != maximum[0] || minimum[1] != maximum[1]) {
    throw std::invalid_argument("CommGrid2D dimensions must agree on all ranks");
  }
  if (rows < 0 || cols < 0) throw std::invalid_argument("CommGrid2D dimensions must be nonnegative");
  const int64_t fixed_size = static_cast<int64_t>(rows == 0 ? 1 : rows) * (cols == 0 ? 1 : cols);
  if (fixed_size > size_ || size_ % fixed_size != 0 || (rows != 0 && cols != 0 && fixed_size != size_)) {
    throw std::invalid_argument("CommGrid2D dimensions must cover every rank exactly once");
  }
  MPI_Dims_create(size_, 2, dimensions);
  rows_ = dimensions[0];
  cols_ = dimensions[1];

  // Report a callback failure collectively, so other ranks do not hang in Allgather.
  std::pair<int, int> coordinates;
  int failed = 0;
  try {
    coordinates = layout ? layout(rank_, rows_, cols_) : std::pair<int, int>{rank_ / cols_, rank_ % cols_};
  } catch (...) {
    failed = 1;
  }
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, world);
  if (any_failed) throw std::invalid_argument("CommGrid2D layout callback failed on a rank");
  row_ = coordinates.first;
  col_ = coordinates.second;

  // Gather the actual mapping, rather than assuming the callback is identical everywhere.
  const int local[2] = {row_, col_};
  std::vector<int> gathered(static_cast<std::size_t>(size_) * 2);
  MPI_Allgather(local, 2, MPI_INT, gathered.data(), 2, MPI_INT, world);
  owners_.assign(size_, -1);
  for (int rank = 0; rank < size_; ++rank) {
    const std::size_t position = static_cast<std::size_t>(rank) * 2;
    const int row = gathered[position];
    const int col = gathered[position + 1];
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
      throw std::invalid_argument("CommGrid2D layout coordinate is outside the grid");
    }
    int& owner = owners_[row * cols_ + col];
    if (owner != -1) throw std::invalid_argument("CommGrid2D layout assigns multiple ranks to a cell");
    owner = rank;
  }

  // Coordinate keys make broadcast roots independent of the source rank layout.
  MPI_Comm_dup(world, &world_);
  MPI_Comm_split(world_, row_, col_, &rowworld_);
  MPI_Comm_split(world_, col_, row_, &colworld_);
}

CommGrid2D::CommGrid2D(CommGrid2D&& rhs) noexcept { *this = std::move(rhs); }

CommGrid2D& CommGrid2D::operator=(CommGrid2D&& rhs) noexcept
{
  if (this != &rhs) {
    Free();
    world_ = std::exchange(rhs.world_, MPI_COMM_NULL);
    rowworld_ = std::exchange(rhs.rowworld_, MPI_COMM_NULL);
    colworld_ = std::exchange(rhs.colworld_, MPI_COMM_NULL);
    rank_ = std::exchange(rhs.rank_, -1);
    size_ = std::exchange(rhs.size_, 0);
    rows_ = std::exchange(rhs.rows_, 0);
    cols_ = std::exchange(rhs.cols_, 0);
    row_ = std::exchange(rhs.row_, -1);
    col_ = std::exchange(rhs.col_, -1);
    owners_ = std::move(rhs.owners_);
  }
  return *this;
}

CommGrid2D::~CommGrid2D() { Free(); }

MPI_Comm CommGrid2D::World() const noexcept { return world_; }

MPI_Comm CommGrid2D::RowWorld() const noexcept { return rowworld_; }

MPI_Comm CommGrid2D::ColWorld() const noexcept { return colworld_; }

int CommGrid2D::Rank() const noexcept { return rank_; }

int CommGrid2D::Size() const noexcept { return size_; }

int CommGrid2D::Rows() const noexcept { return rows_; }

int CommGrid2D::Cols() const noexcept { return cols_; }

int CommGrid2D::Row() const noexcept { return row_; }

int CommGrid2D::Col() const noexcept { return col_; }

int CommGrid2D::RankAt(int row, int col) const
{
  if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
    throw std::out_of_range("CommGrid2D process coordinate is outside the grid");
  }
  return owners_[row * cols_ + col];
}

int CommGrid2D::Owner(int64_t block_row, int64_t block_col) const
{
  if (block_row < 0 || block_col < 0) throw std::out_of_range("Matrix block coordinates must be nonnegative");
  if (world_ == MPI_COMM_NULL) throw std::logic_error("CommGrid2D has been moved from");
  const int row = static_cast<int>(block_row % rows_);
  const int col = static_cast<int>(block_col % cols_);
  return RankAt(row, col);
}

void CommGrid2D::Free() noexcept
{
  if (rowworld_ != MPI_COMM_NULL) MPI_Comm_free(&rowworld_);
  if (colworld_ != MPI_COMM_NULL) MPI_Comm_free(&colworld_);
  if (world_ != MPI_COMM_NULL) MPI_Comm_free(&world_);
}

}  // namespace spcraft
