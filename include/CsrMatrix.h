#pragma once

#include <cstdint>
#include <tuple>

namespace spcraft
{

/**
 * @brief Compressed Sparse Row (CSR) matrix storage.
 *
 * `row_ptr` holds `m+1` row pointers into the `col_id`/`val` arrays. The row
 * pointers are offsets in `[0, nnz]`, so they (and `nnz`) use the offset type
 * `OT`, while the column indices and dimensions use the index type `IT`.
 *
 * Move-only: copy construction and assignment are deleted; use `Clone()` for an
 * explicit deep copy. A matrix either owns its storage (`memowned == true`) or
 * is a non-owning view over externally managed buffers (`memowned == false`).
 *
 * @tparam IT Index type for column indices and the dimensions `m` and `n`.
 * @tparam NT Numeric type of the stored values.
 * @tparam OT Offset type for the row pointers and the nonzero count. Defaults
 *            to `IT`; set it wider (e.g. `int64_t`) when the matrix can hold
 *            more nonzeros than `IT` can address.
 */
template <class IT, class NT, class OT = IT>
class CsrMatrix
{
  static std::tuple<OT*, IT*, NT*> SafeAllocate(IT m, OT nnz);
  static void SafeDelete(bool memowned, OT* row_ptr, IT* col_id, NT* val);

 public:
  OT* row_ptr = nullptr;  //!< row pointers, size m+1
  IT* col_id = nullptr;   //!< column indices, size nnz
  NT* val = nullptr;      //!< numerical values, size nnz

  OT nnz = 0;  //!< number of nonzeros
  IT m = 0;    //!< number of rows
  IT n = 0;    //!< number of columns

  bool memowned = true;  //!< owns the storage (views opt out)

  //! Empty matrix (owns nothing yet).
  CsrMatrix() = default;

  /**
   * @brief Wrap externally managed buffers as a non-owning view.
   *
   * @param[in] row_ptr_ Row pointers (m+1 entries).
   * @param[in] col_id_  Column indices (nnz entries).
   * @param[in] val_     Values (nnz entries).
   * @param[in] nnz_     Number of nonzeros.
   * @param[in] m_       Number of rows.
   * @param[in] n_       Number of columns.
   */
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

  /**
   * @brief Allocate owned, uninitialized storage for `require_nnz` nonzeros.
   *
   * @param[in] require_nnz Number of nonzeros to allocate.
   * @param[in] nRows        Number of rows.
   * @param[in] nCols        Number of columns.
   */
  void Allocate(OT require_nnz, IT nRows, IT nCols);

  /** @brief Explicit deep copy (copy construction is deleted). */
  CsrMatrix<IT, NT, OT> Clone() const;

  /** @brief Null every member without freeing; releases ownership. */
  void Reset();
};

}  // namespace spcraft

#include "CsrMatrix-inl.h"
