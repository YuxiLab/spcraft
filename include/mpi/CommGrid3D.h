#pragma once

#include <cstdint>
#include <memory>

#include "mpi/CommGrid2D.h"

namespace spcraft
{

/**
 * @brief Layer-major grid of square 2D planes with fibers joining equal coordinates.
 * Construction is collective on world. Its size must be layers * side * side.
 * Source rank = layer * side * side + row * side + column. Plane() owns the
 * row/column communicators; FiberWorld() ranks equal layer coordinates.
 * A block (i,k) belongs to layer k % Layers(), plane coordinate (i % side,k % side).
 * B block (k,j) belongs to layer k % Layers(), plane coordinate (k % side,j % side).
 * Local indices retain the plane's full 2D cyclic indexing; unowned inner blocks
 * have no entries. Thus inputs are partitioned across layers without replication.
 * All communicator handles are borrowed. Release every grid owner before MPI_Finalize.
 */
class CommGrid3D
{
 public:
  using PlanePtr = std::shared_ptr<const CommGrid2D>;

 private:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  MPI_Comm world_ = MPI_COMM_NULL;  //!< Owned duplicate of the source communicator.
  MPI_Comm fiber_ = MPI_COMM_NULL;  //!< Owned communicator across corresponding plane cells.
  PlanePtr plane_{};                //!< Shared owner of this layer's square 2D grid.
  int layers_ = 0;                  //!< Number of layers.
  int layer_ = -1;                  //!< This rank's layer coordinate.

 public:
  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  CommGrid3D(MPI_Comm world, int layers);

  CommGrid3D(const CommGrid3D&) = delete;

  CommGrid3D& operator=(const CommGrid3D&) = delete;

  CommGrid3D(CommGrid3D&& rhs) noexcept;

  CommGrid3D& operator=(CommGrid3D&& rhs) noexcept;

  ~CommGrid3D();

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  MPI_Comm World() const noexcept;

  MPI_Comm FiberWorld() const noexcept;

  const PlanePtr& Plane() const noexcept;

  int Layers() const noexcept;

  int Layer() const noexcept;

  //! Source rank at a valid (row, column, layer) coordinate.
  int RankAt(int row, int col, int layer) const;

  //! Source ranks owning A(i,k) and B(k,j), respectively, in block coordinates.
  int AOwner(int64_t block_row, int64_t inner_block) const;

  int BOwner(int64_t inner_block, int64_t block_col) const;

 private:
  void Free() noexcept;
};

}  // namespace spcraft
