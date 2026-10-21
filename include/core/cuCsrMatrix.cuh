#pragma once

#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

#include "core/CsrMatrix.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Compressed Sparse Row (CSR) matrix stored in CUDA device memory.
 * Link SPCraft for float/double/bool and signed (IT, OT) widths
 * (32, 32), (32, 64), or (64, 64).
 *
 * Owns cudaMalloc storage, or borrows device buffers through the pointer
 * constructor. Buffers are only dereferenceable on the device. Copying is
 * explicit through Clone(); moves transfer ownership and reset the source.
 *
 * Host code uploads through the host-matrix constructor and downloads with
 * ToHost(). View() borrows the same device buffers for kernel entry points.
 *
 * Allocation and transfer failures throw because constructors cannot return
 * CUDA error codes. Destruction ignores CUDA errors.
 */
template <class IT, class NT, class OT = IT>
class cuCsrMatrix
{
 public:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  OT* row_ptr = nullptr;  //!< row pointers, size m+1, device memory
  IT* col_id = nullptr;   //!< column indices, size nnz, device memory
  NT* val = nullptr;      //!< numerical values, size nnz, device memory
  OT nnz = 0;             //!< number of nonzeros
  IT m = 0;               //!< number of rows
  IT n = 0;               //!< number of columns
  bool memowned = true;   //!< owns the storage (views opt out)

  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  //! Empty matrix (owns nothing yet).
  cuCsrMatrix() noexcept;

  //! Wrap externally managed device buffers as a non-owning view.
  cuCsrMatrix(OT* row_ptr_, IT* col_id_, NT* val_, OT nnz_, IT m_, IT n_);

  //! Upload a host CSR matrix into owned device storage.
  cuCsrMatrix(const SPCSR& host);

  //! Disable copy construction.
  cuCsrMatrix(const cuCsrMatrix& rhs) = delete;

  //! Disable copy assignment.
  cuCsrMatrix& operator=(const cuCsrMatrix& rhs) = delete;

  //! Transfer storage and ownership, then reset the source.
  cuCsrMatrix(cuCsrMatrix&& rhs) noexcept;

  //! Free owned storage, transfer the source, and reset it; self-move is a no-op.
  cuCsrMatrix& operator=(cuCsrMatrix&& rhs) noexcept;

  //! Frees owned device storage.
  ~cuCsrMatrix();

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  //! Replace storage after allocation succeeds; zero nonzeros retain the shape.
  void Allocate(OT require_nnz, IT nRows, IT nCols);

  //! Create an owned device-to-device copy, including an empty matrix's shape.
  cuCsrMatrix Clone() const;

  //! Clear members without freeing; calling this on live owned storage leaks it.
  void Reset();

  //! Download into a newly allocated host CSR matrix.
  [[nodiscard]] SPCSR ToHost() const;

  //! Borrow device buffers for kernels; the view cannot outlive their storage.
  [[nodiscard]] SPCSR View() const;

 private:
  /*****************************************************************************
   *                              Static Members                               *
   ****************************************************************************/
  //! Allocate zeroed device buffers; free partial allocations before throwing.
  [[nodiscard]] static std::tuple<OT*, IT*, NT*> SafeAllocate(IT m, OT nnz);

  //! Free owned buffers; callers must clear or overwrite their member pointers.
  static void SafeDelete(bool memowned, OT* row_ptr, IT* col_id, NT* val);
};

}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
