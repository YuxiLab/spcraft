#pragma once

#include <cstdint>
#include <string>
#include <tuple>

#include "CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Coordinate (COO) matrix storage format.
 */
template <class IT, class NT, class OT = IT>
class CooMatrix
{
  /***************************
   * static function
   * *************************/
  [[nodiscard]] static std::tuple<IT*, IT*, NT*> SafeAllocate(OT nnz);
  static void SafeDelete(bool memowned, IT* row_id, IT* col_id, NT* val);

 public:
  /***************************
   * data member
   * *************************/
  IT* row_id = nullptr;  //!< row indices, size nnz
  IT* col_id = nullptr;  //!< column indices, size nnz
  NT* val = nullptr;     //!< numerical values, size nnz
  OT nnz = 0;            //!< number of nonzeros
  IT m = 0;              //!< number of rows
  IT n = 0;              //!< number of columns
  bool memowned = true;  //!< owns the storage (views opt out)

  /***************************
   * function member
   * *************************/
  //! Empty matrix (owns nothing yet).
  CooMatrix() = default;
  //! Wrap externally managed buffers as a non-owning view.
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
  //! explicitly allocate the memory
  void Allocate(OT require_nnz, IT nRows, IT nCols);
  //! clone a new instance
  CooMatrix<IT, NT, OT> Clone() const;
  //! Null every member without freeing; releases ownership.
  void Reset();

  //! Read a Matrix Market (.mtx) file into COO format.
  [[nodiscard]] static CooMatrix<IT, NT, OT> FromMatrixMarket(const std::string& filename);
  /**
   * @brief Convert COO matrix to CSR matrix format.
   */
  CsrMatrix<IT, NT, OT> ToCsr() const;
};

}  // namespace spcraft

#include "CooMatrix-inl.h"
