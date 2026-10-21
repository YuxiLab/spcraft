#pragma once

#include <cstdint>
#include <tuple>
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Compressed Sparse Column (CSC) matrix format.
 * Link SPCraft for float, double, or bool values with (IT, OT) widths
 * (32, 32), (32, 64), or (64, 64), using signed fixed-width integer types.
 */
template <class IT, class NT, class OT = IT>
class CscMatrix
{
 public:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  OT* col_ptr = nullptr;  //!< column pointers, size n+1
  IT* row_id = nullptr;   //!< row indices, size nnz
  NT* val = nullptr;      //!< numerical values, size nnz
  OT nnz = 0;             //!< number of nonzeros
  IT m = 0;               //!< number of rows
  IT n = 0;               //!< number of columns
  bool memowned = true;   //!< owns the storage (views opt out)

  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  //! Empty matrix (owns nothing yet).
  CscMatrix() noexcept;

  //! Wrap externally managed buffers as a non-owning view.
  CscMatrix(OT* col_ptr_, IT* row_id_, NT* val_, OT nnz_, IT m_, IT n_);

  //! Move constructor (transfers ownership).
  CscMatrix(SPCSC&& rhs) noexcept;

  //! Move assignment (frees the old storage, takes the new).
  SPCSC& operator=(SPCSC&& rhs) noexcept;

  //! Disable copy construction.
  CscMatrix(const SPCSC& rhs) = delete;

  //! Disable copy assignment.
  CscMatrix& operator=(const SPCSC& rhs) = delete;

  //! Frees owned storage.
  ~CscMatrix();

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  //! clone a new instance
  SPCSC Clone() const;

  //! explicitly allocate the memory
  void Allocate(OT require_nnz, IT nRows, IT nCols);

  //! Null every member without freeing; releases ownership.
  void Reset();

 private:
  /*****************************************************************************
   *                              Static Members                               *
   ****************************************************************************/
  SPND static std::tuple<OT*, IT*, NT*> SafeAllocate(IT n, OT nnz);

  static void SafeDelete(bool memowned, OT* col_ptr, IT* row_id, NT* val);
};

}  // namespace spcraft
