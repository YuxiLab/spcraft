#pragma once

#include "core/CsrMatrix.h"
#include <cstdint>
#include <memory>

#include "mpi/CommGrid2D.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Distributed CSR matrix.
 */
template <class IT, class NT, class OT = IT>
class DistSpCSR
{
 public:
  using MatrixType = CsrMatrix<IT, NT, OT>;
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
  DistSpCSR() noexcept;

  //! Create a dimensioned empty local matrix; no redistribution is performed.
  DistSpCSR(GridPtr grid, IT global_m, IT global_n);

  //! Take local storage whose dimensions match this rank's contiguous partition.
  DistSpCSR(GridPtr grid, MatrixType&& data, IT global_m, IT global_n);

  DistSpCSR(const DistSpCSR&) = delete;

  DistSpCSR& operator=(const DistSpCSR&) = delete;

  DistSpCSR(DistSpCSR&& rhs) noexcept;

  DistSpCSR& operator=(DistSpCSR&& rhs) noexcept;

  ~DistSpCSR();

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  //! Allocate replacement local storage before releasing the old buffers.
  void Allocate(OT count);

  void Allocate(OT count, IT global_m, IT global_n);

  [[nodiscard]] DistSpCSR Clone() const;

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
