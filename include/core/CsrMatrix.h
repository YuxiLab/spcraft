#pragma once

#include <cstdint>
#include <tuple>

namespace spcraft
{

/**
 * @brief Compressed Sparse Row (CSR) matrix format.
 */
template <class IT, class NT, class OT = IT>
class CsrMatrix
{
  /***************************
   * static function
   * *************************/
  [[nodiscard]] static std::tuple<OT*, IT*, NT*> SafeAllocate(IT m, OT nnz);
  static void SafeDelete(bool memowned, OT* row_ptr, IT* col_id, NT* val);

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

  /***************************
   * function member
   * *************************/
  //! Empty matrix (owns nothing yet).
  CsrMatrix() = default;
  //! Wrap externally managed buffers as a non-owning view.
  CsrMatrix(OT* row_ptr_, IT* col_id_, NT* val_, OT nnz_, IT m_, IT n_);
  //! Disable copy construction.
  CsrMatrix(const CsrMatrix<IT, NT, OT>& rhs) = delete;
  //! Disable copy assignment.
  CsrMatrix& operator=(const CsrMatrix<IT, NT, OT>& rhs) = delete;
  //! Move constructor (transfers ownership).
  CsrMatrix(CsrMatrix<IT, NT, OT>&& rhs) noexcept;
  //! Move assignment (frees the old storage, takes the new).
  CsrMatrix<IT, NT, OT>& operator=(CsrMatrix<IT, NT, OT>&& rhs) noexcept;
  //! Frees owned storage.
  ~CsrMatrix();
  //! explicitly allocate the memory
  void Allocate(OT require_nnz, IT nRows, IT nCols);
  //! clone a new instance
  CsrMatrix<IT, NT, OT> Clone() const;
  //! Null every member without freeing; releases ownership.
  void Reset();
};

}  // namespace spcraft

#include "core/CsrMatrix-inl.h"
