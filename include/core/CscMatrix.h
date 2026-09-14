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
  // clang-format off
  //! Wrap externally managed buffers as a non-owning view.
  CscMatrix(OT* col_ptr_, IT* row_id_, NT* val_, OT nnz_, IT m_, IT n_): col_ptr(col_ptr_), row_id(row_id_), val(val_), nnz(nnz_), m(m_), n(n_), memowned(false){}
  //! Move constructor (transfers ownership).
  CscMatrix(CscMatrix<IT, NT, OT>&& rhs) noexcept : col_ptr(rhs.col_ptr), row_id(rhs.row_id),val(rhs.val),nnz(rhs.nnz),m(rhs.m),n(rhs.n), memowned(rhs.memowned){rhs.Reset();}
  // clang-format on
  //! Move assignment (frees the old storage, takes the new).
  CscMatrix<IT, NT, OT>& operator=(CscMatrix<IT, NT, OT>&& rhs) noexcept;
  //! Disable copy construction.
  CscMatrix(const CscMatrix<IT, NT, OT>& rhs) = delete;
  //! Disable copy assignment.
  CscMatrix& operator=(const CscMatrix<IT, NT, OT>& rhs) = delete;
  //! Frees owned storage.
  ~CscMatrix();
  //! clone a new instance
  CscMatrix<IT, NT, OT> Clone() const;
  //! explicitly allocate the memory
  void Allocate(OT require_nnz, IT nRows, IT nCols);
  //! Null every member without freeing; releases ownership.
  void Reset();

 private:
  /***************************
   * static function
   * *************************/
  [[nodiscard]] static std::tuple<OT*, IT*, NT*> SafeAllocate(IT n, OT nnz);
  static void SafeDelete(bool memowned, OT* col_ptr, IT* row_id, NT* val);
};

}  // namespace spcraft

#include "core/CscMatrix-inl.h"
