#pragma once

#include <cstdint>
#include <tuple>
#include <string>
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Compressed Sparse Row (CSR) matrix format.
 * Link SPCraft for float, double, or bool values with (IT, OT) widths
 * (32, 32), (32, 64), or (64, 64), using signed fixed-width integer types.
 */
SP_MAT_TEMP_DEFAULT
class CsrMatrix
{
 public:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  OT* row_ptr = nullptr;  //!< row pointers, size m+1
  IT* col_id = nullptr;   //!< column indices, size nnz
  NT* val = nullptr;      //!< numerical values, size nnz
  OT nnz = 0;             //!< number of nonzeros
  IT m = 0;               //!< number of rows
  IT n = 0;               //!< number of columns
  bool memowned = true;   //!< owns the storage (views opt out)

  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  //! Empty matrix (owns nothing yet).
  CsrMatrix() noexcept;

  //! Wrap externally managed buffers as a non-owning view.
  CsrMatrix(OT* row_ptr_, IT* col_id_, NT* val_, OT nnz_, IT m_, IT n_);

  CsrMatrix(OT nnz_, IT m_, IT n_);

  //! Move constructor (transfers ownership).
  CsrMatrix(SPCSR&& rhs) noexcept;

  //! Move assignment (frees the old storage, takes the new).
  SPCSR& operator=(SPCSR&& rhs) noexcept;

  //! Frees owned storage.
  ~CsrMatrix();

  //! Disable copy construction.
  CsrMatrix(const SPCSR& rhs) = delete;

  //! Disable copy assignment.
  CsrMatrix& operator=(const SPCSR& rhs) = delete;

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  //! clone a new CsrMatrix explicitly.
  SPCSR Clone() const;

  //! explicitly allocate the memory
  void Allocate(OT require_nnz, IT nRows, IT nCols);

  // Null the pointer and set Zero, not responsible for freeing memory.
  void Reset() noexcept;

 private:
  /*****************************************************************************
   *                              Static Members                               *
   ****************************************************************************/
  // tedious function with all boundary check.
  SPND static std::tuple<OT*, IT*, NT*> SafeAllocate(OT required_nnz, IT rows, IT columns);

  // delete the memory if owned.
  static void SafeDelete(bool owned, OT* row_ptr, IT* col_id, NT* val) noexcept;
};

}  // namespace spcraft
