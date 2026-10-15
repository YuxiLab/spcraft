#pragma once

#include "core/DcscMatrix.h"
#include <cstdint>
#include <memory>

#include "mpi/CommGrid2D.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Distributed DCSC storage with shared grid ownership and move-only local storage.
 * Each rank owns one contiguous local submatrix. Global m and n determine the
 * partition through the process grid. Clone copies local buffers and shares
 * the grid. The local matrix's memowned flag controls ownership of its buffers.
 * Construction and Allocate are local operations; all ranks supply the same
 * global dimensions and grid mapping. Populate or redistribute entries separately.
 * Release all shared grid owners before MPI_Finalize.
 * Link SPCraft with MPI enabled for float, double, or bool and signed (IT, OT)
 * widths (32, 32), (32, 64), or (64, 64).
 */
SP_MAT_TEMP_DEFAULT
class DistSpDCSC
{
 public:
  using MatrixType = DcscMatrix<IT, NT, OT>;
  using GridPtr = std::shared_ptr<const CommGrid2D>;

  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  GridPtr commgrid{};  //!< Shared ownership of the process grid.
  MatrixType local{};  //!< Native local storage with its own buffer ownership.
  IT m = 0;            //!< Global row count.
  IT n = 0;            //!< Global column count.

  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  DistSpDCSC() noexcept;

  //! Create a dimensioned empty local matrix; no redistribution is performed.
  DistSpDCSC(GridPtr grid, IT global_m, IT global_n);

  //! Take local storage whose dimensions match this rank's contiguous partition.
  DistSpDCSC(GridPtr grid, MatrixType&& data, IT global_m, IT global_n);

  DistSpDCSC(const DistSpDCSC&) = delete;

  DistSpDCSC& operator=(const DistSpDCSC&) = delete;

  DistSpDCSC(DistSpDCSC&& rhs) noexcept;

  DistSpDCSC& operator=(DistSpDCSC&& rhs) noexcept;

  ~DistSpDCSC();

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  //! Nonempty allocations require the number of stored local columns.
  void Allocate(OT count, IT stored_columns);

  void Allocate(OT count, IT global_m, IT global_n, IT stored_columns);

  SPND DistSpDCSC Clone() const;

  //! Release ownership without freeing local buffers, as in the local containers.
  void Reset() noexcept;

  IT LocalRows() const;

  IT LocalCols() const;

  IT GlobalRow(IT local_row) const;

  IT GlobalCol(IT local_col) const;

  //! Reject a global row/column belonging to another process row/column.
  IT LocalRow(IT global_row) const;

  IT LocalCol(IT global_col) const;

  //! World rank owning a global matrix entry.
  int Owner(IT global_row, IT global_col) const;

  bool Owns(IT global_row, IT global_col) const;

  //! Collective sum of local nnz; the total must fit int64_t, independently of OT.
  int64_t GlobalNnz() const;
};

}  // namespace spcraft
