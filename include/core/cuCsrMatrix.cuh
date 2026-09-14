#pragma once

#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

#include "core/CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Compressed Sparse Row (CSR) matrix stored in CUDA device memory.
 *
 * Mirrors CsrMatrix, but the owned storage comes from cudaMalloc rather than
 * std::calloc, so the buffers are only dereferenceable on the device. Host code
 * moves data across with FromHost() and ToHost(); kernels consume View(), which
 * hands out a non-owning CsrMatrix wrapping the same device pointers.
 *
 * Allocation and transfer failures throw (a container constructor has no error
 * code to return); the destructor swallows them.
 */
template <class IT, class NT, class OT = IT>
class cuCsrMatrix
{
 public:
  /*************************************
   *             Data Members
   *************************************/
  OT* row_ptr = nullptr;  //!< row pointers, size m+1, device memory
  IT* col_id = nullptr;   //!< column indices, size nnz, device memory
  NT* val = nullptr;      //!< numerical values, size nnz, device memory
  OT nnz = 0;             //!< number of nonzeros
  IT m = 0;               //!< number of rows
  IT n = 0;               //!< number of columns
  bool memowned = true;   //!< owns the storage (views opt out)

  /*************************************
   *             constructor
   *************************************/
  //! Empty matrix (owns nothing yet).
  cuCsrMatrix() = default;
  //! Wrap externally managed device buffers as a non-owning view.
  cuCsrMatrix(OT* row_ptr_, IT* col_id_, NT* val_, OT nnz_, IT m_, IT n_);
  cuCsrMatrix(const CsrMatrix<IT, NT, OT>& host);  //!< Upload a host CSR matrix into device memory.
  //! Disable copy construction.
  cuCsrMatrix(const cuCsrMatrix<IT, NT, OT>& rhs) = delete;
  //! Disable copy assignment.
  cuCsrMatrix& operator=(const cuCsrMatrix<IT, NT, OT>& rhs) = delete;
  //! Move constructor (transfers ownership).
  cuCsrMatrix(cuCsrMatrix<IT, NT, OT>&& rhs) noexcept;
  //! Move assignment (frees the old storage, takes the new).
  cuCsrMatrix<IT, NT, OT>& operator=(cuCsrMatrix<IT, NT, OT>&& rhs) noexcept;
  //! Frees owned device storage.
  ~cuCsrMatrix();

  /*************************************
   *             member functions
   *************************************/
  //! explicitly allocate the device memory
  void Allocate(OT require_nnz, IT nRows, IT nCols);
  //! clone a new instance (device to device)
  cuCsrMatrix<IT, NT, OT> Clone() const;
  //! Null every member without freeing; releases ownership.
  void Reset();

  //! Download into a newly allocated host CSR matrix.
  [[nodiscard]] CsrMatrix<IT, NT, OT> ToHost() const;
  //! Non-owning CsrMatrix over the same device buffers, for kernel entry points.
  [[nodiscard]] CsrMatrix<IT, NT, OT> View() const;

 private:
  /***************************
   * static function
   * *************************/
  //! Upload a host CSR matrix into freshly allocated device memory.
  [[nodiscard]] static std::tuple<OT*, IT*, NT*> SafeAllocate(IT m, OT nnz);
  static void SafeDelete(bool memowned, OT* row_ptr, IT* col_id, NT* val);
};

}  // namespace spcraft

#include "core/cuCsrMatrix-inl.cuh"

#endif  // SPCRAFT_USE_CUDA
