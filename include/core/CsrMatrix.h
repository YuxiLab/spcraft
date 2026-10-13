#pragma once

#include <cstdint>
#include <tuple>
#include <string>
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Compressed Sparse Row (CSR) matrix format.
 */
SP_MAT_TEMP_DEFAULT
class CsrMatrix
{
 public:
  /***************************
   * data member
   * *************************/
  OT* row_ptr = nullptr;  //!< row pointers, size m+1
  IT* col_id = nullptr;   //!< column indices, size nnz
  NT* val = nullptr;      //!< numerical values, size nnz
  OT nnz = 0;             //!< number of nonzeros
  IT m = 0;               //!< number of rows
  IT n = 0;               //!< number of columns
  bool memowned = true;   //!< owns the storage (views opt out)

  /*************************************
   *             constructor
   *************************************/
  //! Empty matrix (owns nothing yet).
  CsrMatrix() = default;

  //! Wrap externally managed buffers as a non-owning view.
  CsrMatrix(OT* row_ptr_, IT* col_id_, NT* val_, OT nnz_, IT m_, IT n_)
      : row_ptr(row_ptr_), col_id(col_id_), val(val_), nnz(nnz_), m(m_), n(n_), memowned(false)
  {
  }

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

  /*************************************
   *             Data Members
   *************************************/
  //! clone a new CsrMatrix explicitly.
  SPCSR Clone() const;

  //! explicitly allocate the memory
  void Allocate(OT require_nnz, IT nRows, IT nCols);

  // Null the pointer and set Zero, not responsible for freeing memory.
  void Reset() noexcept;

 private:
  /*************************************
   *             Static Members
   *************************************/
  // tedious function with all boundary check.
  SPND static std::tuple<OT*, IT*, NT*> SafeAllocate(OT required_nnz, IT rows, IT columns);

  // delete the memory if owned.
  static void SafeDelete(bool owned, OT* row_ptr, IT* col_id, NT* val) noexcept;
};

}  // namespace spcraft

#include "core/CsrMatrix-inl.h"
