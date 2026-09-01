#pragma once

#include <cstdint>
#include <string>
#include <tuple>

#include "CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Coordinate (COO) matrix storage format.
 *
 * `row_id`, `col_id`, and `val` arrays of size `nnz`.
 *
 * Move-only: copy construction and assignment are deleted; use `Clone()` for an
 * explicit deep copy. A matrix either owns its storage (`memowned == true`) or
 * is a non-owning view over externally managed buffers (`memowned == false`).
 *
 * @tparam IT Index type for row and column indices and dimensions `m` and `n`.
 * @tparam NT Numeric type of the stored values.
 * @tparam OT Offset type for nonzeros count `nnz`. Defaults to `IT`.
 */
template <class IT, class NT, class OT = IT>
class CooMatrix
{
  static std::tuple<IT*, IT*, NT*> SafeAllocate(OT nnz);
  static void SafeDelete(bool memowned, IT* row_id, IT* col_id, NT* val);

 public:
  IT* row_id = nullptr;  //!< row indices, size nnz
  IT* col_id = nullptr;  //!< column indices, size nnz
  NT* val = nullptr;     //!< numerical values, size nnz

  OT nnz = 0;  //!< number of nonzeros
  IT m = 0;    //!< number of rows
  IT n = 0;    //!< number of columns

  bool memowned = true;  //!< owns the storage (views opt out)

  //! Empty matrix (owns nothing yet).
  CooMatrix() = default;

  /**
   * @brief Wrap externally managed buffers as a non-owning view.
   */
  CooMatrix(IT* row_id_, IT* col_id_, NT* val_, OT nnz_, IT m_, IT n_);

  //! Disable copy construction.
  CooMatrix(const CooMatrix<IT, NT, OT>& rhs) = delete;
  //! Disable copy assignment.
  CooMatrix& operator=(const CooMatrix<IT, NT, OT>& rhs) = delete;
  //! Move constructor (transfers ownership).
  CooMatrix(CooMatrix<IT, NT, OT>&& rhs) noexcept;
  //! Move assignment (frees the old storage, takes the new).
  CooMatrix<IT, NT, OT>& operator=(CooMatrix<IT, NT, OT>&& rhs) noexcept;
  //! Frees owned storage.
  ~CooMatrix();

  /**
   * @brief Allocate owned, uninitialized storage for `require_nnz` nonzeros.
   */
  void Allocate(OT require_nnz, IT nRows, IT nCols);

  /** @brief Explicit deep copy (copy construction is deleted). */
  CooMatrix<IT, NT, OT> Clone() const;

  /** @brief Null every member without freeing; releases ownership. */
  void Reset();

  /**
   * @brief Convert COO matrix to CSR matrix format.
   */
  CsrMatrix<IT, NT, OT> ToCsr() const;

  /**
   * @brief Read matrix from Matrix Market (.mtx) file into COO format.
   */
  static CooMatrix<IT, NT, OT> FromMatrixMarket(const std::string& filename);
};

}  // namespace spcraft

#include "CooMatrix-inl.h"
