#pragma once

#include <mpi.h>

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Process ownership and row/column communicators for a 2D block-cyclic layout.
 * Construction is collective on an intracommunicator. Zero dimensions choose a
 * balanced grid (square when possible); positive dimensions constrain its shape.
 * A layout callback maps each source rank to a unique (row, column), and is
 * called once on that rank. All ranks must supply the same requested dimensions.
 * Block (i, j) belongs to process coordinates (i % Rows(), j % Cols()). For a
 * contiguous partition with one block per process, these are simply (i, j).
 * A, B, and C use this same ownership rule; partitioning and moving their
 * buffers remain the caller's responsibility.
 *
 * World ranks preserve the source communicator's order. RowWorld() ranks equal
 * column coordinates, and ColWorld() ranks equal row coordinates, even with a
 * custom mapping. Handles are borrowed: do not free them. Destroy the grid on
 * all participating ranks before MPI_Finalize; moves transfer its ownership.
 */
class CommGrid2D
{
 public:
  using RankLayout = std::function<std::pair<int, int>(int rank, int rows, int cols)>;

 private:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  MPI_Comm world_ = MPI_COMM_NULL;     //!< Owned duplicate of the source communicator.
  MPI_Comm rowworld_ = MPI_COMM_NULL;  //!< Owned communicator with ranks ordered by column.
  MPI_Comm colworld_ = MPI_COMM_NULL;  //!< Owned communicator with ranks ordered by row.
  int rank_ = -1;                      //!< Source communicator rank.
  int size_ = 0;                       //!< Total participating ranks.
  int rows_ = 0;                       //!< Process row count.
  int cols_ = 0;                       //!< Process column count.
  int row_ = -1;                       //!< This rank's process row.
  int col_ = -1;                       //!< This rank's process column.
  std::vector<int> owners_{};          //!< Source rank at each row-major process coordinate.

 public:
  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  explicit CommGrid2D(MPI_Comm world = MPI_COMM_WORLD, int rows = 0, int cols = 0);

  CommGrid2D(MPI_Comm world, int rows, int cols, RankLayout layout);

  CommGrid2D(const CommGrid2D&) = delete;

  CommGrid2D& operator=(const CommGrid2D&) = delete;

  CommGrid2D(CommGrid2D&& rhs) noexcept;

  CommGrid2D& operator=(CommGrid2D&& rhs) noexcept;

  ~CommGrid2D();

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  SPND MPI_Comm World() const noexcept;

  SPND MPI_Comm RowWorld() const noexcept;

  SPND MPI_Comm ColWorld() const noexcept;

  SPND int Rank() const noexcept;

  SPND int Size() const noexcept;

  SPND int Rows() const noexcept;

  SPND int Cols() const noexcept;

  SPND int Row() const noexcept;

  SPND int Col() const noexcept;

  //! Source/world rank at process coordinates; rejects coordinates outside the grid.
  SPND int RankAt(int row, int col) const;

  //! Source/world rank owning a nonnegative matrix block coordinate.
  SPND int Owner(int64_t block_row, int64_t block_col) const;

 private:
  void Free() noexcept;
};

}  // namespace spcraft
