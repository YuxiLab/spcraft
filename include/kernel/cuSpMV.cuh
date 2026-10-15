#pragma once

#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "core/CsrMatrix.h"
#include "core/DenseVector.h"
#include "semiring/SemiRing.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{
namespace detail
{

template <class SemiRing, typename IT, typename NT, typename OT, int kBlockSize>
struct SpmvCudaBlockPerRowImpl {
  /*****************************************************************************
   *                              Static Members                               *
   ****************************************************************************/
  static cudaError_t Run(const SPCSR& matrix, const SPDVEC& x, SPDVEC& y, cudaStream_t stream);
};

}  // namespace detail

/**
 * @brief Compute y = A * x for a CSR matrix using one CUDA block per row.
 *
 * IT is the column-index and row-count type, NT is the numerical
 * type, and OT is the row-offset type. The CSR matrix must wrap
 * device-memory buffers. The launch is asynchronous with respect to the host.
 *
 * @param matrix CSR matrix view whose buffers reside in device memory.
 * @param x Dense input vector view whose buffer resides in device memory.
 * @param y Dense output vector view whose buffer resides in device memory.
 * @tparam SemiRing Scalar addition, multiplication, and identity operations.
 * @tparam kBlockSize Power-of-two CUDA block size in the range [1, 1024].
 * @param stream CUDA stream used for the kernel launch.
 */
template <class SemiRing, typename IT, typename NT, typename OT = IT, int kBlockSize = 256>
cudaError_t SpmvCudaBlockPerRow(const SPCSR& matrix, const SPDVEC& x, SPDVEC& y,
                                cudaStream_t stream = nullptr)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "SpMV semiring value type must match the matrix value type");
  if constexpr (std::is_signed_v<IT>) {
    if (matrix.m < IT{0} || matrix.n < IT{0}) return cudaErrorInvalidValue;
  }
  if constexpr (std::is_signed_v<OT>) {
    if (matrix.nnz < OT{0}) return cudaErrorInvalidValue;
  }
  if (x.n != matrix.n || y.n != matrix.m) return cudaErrorInvalidValue;
  if (matrix.m == IT{0}) return cudaSuccess;
  if (matrix.row_ptr == nullptr || matrix.col_id == nullptr || matrix.val == nullptr || x.val == nullptr ||
      y.val == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }

  return detail::SpmvCudaBlockPerRowImpl<SemiRing, IT, NT, OT, kBlockSize>::Run(matrix, x, y, stream);
}

}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
