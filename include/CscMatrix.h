#pragma once

#include <cstdint>
#include <tuple>

namespace spcraft
{

/**
 * @brief Compressed Sparse Column (CSC) matrix format.
 */
template <class IT, class NT, class OT = IT>
class CscMatrix
{
  /***************************
   * static function
   * *************************/
  [[nodiscard]] static std::tuple<OT*, IT*, NT*> SafeAllocate(IT n, OT nnz);
  static void SafeDelete(bool memowned, OT* col_ptr, IT* row_id, NT* val);

 public:
  /***************************
   * data member
   * *************************/
  OT* col_ptr = nullptr;  //!< column pointers, size n+1
  IT* row_id = nullptr;   //!< row indices, size nnz
  NT* val = nullptr;      //!< numerical values, size nnz
  OT nnz = 0;             //!< number of nonzeros
  IT m = 0;               //!< number of rows
  IT n = 0;               //!< number of columns
  bool memowned = true;   //!< owns the storage (views opt out)

  /***************************
   * memeber function
   * *************************/
  //! Empty matrix (owns nothing yet).
  CscMatrix() = default;
  //! Wrap externally managed buffers as a non-owning view.
  CscMatrix(OT* col_ptr_, IT* row_id_, NT* val_, OT nnz_, IT m_, IT n_);
  //! Disable copy construction.
  CscMatrix(const CscMatrix<IT, NT, OT>& rhs) = delete;
  //! Disable copy assignment.
  CscMatrix& operator=(const CscMatrix<IT, NT, OT>& rhs) = delete;
  //! Move constructor (transfers ownership).
  CscMatrix(CscMatrix<IT, NT, OT>&& rhs) noexcept;
  //! Move assignment (frees the old storage, takes the new).
  CscMatrix<IT, NT, OT>& operator=(CscMatrix<IT, NT, OT>&& rhs) noexcept;
  //! Frees owned storage.
  ~CscMatrix();
  //! explicitly allocate the memory
  void Allocate(OT require_nnz, IT nRows, IT nCols);
  //! clone a new instance
  CscMatrix<IT, NT, OT> Clone() const;
  //! Null every member without freeing; releases ownership.
  void Reset();
};

}  // namespace spcraft

#include "CscMatrix-inl.h"
