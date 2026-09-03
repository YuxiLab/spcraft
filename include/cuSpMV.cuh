#pragma once

#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "CsrMatrix.h"
#include "DenseVector.h"
#include "SemiRing.h"

namespace spcraft
{
namespace detail
{

template <class SemiRing, typename IT, typename NT, typename OT, int kBlockSize>
struct SpmvCudaBlockPerRowImpl {
  static cudaError_t Run(const CsrMatrix<IT, NT, OT>& matrix, const DenseVector<IT, NT>& x,
                         DenseVector<IT, NT>& y, cudaStream_t stream);
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
cudaError_t SpmvCudaBlockPerRow(const CsrMatrix<IT, NT, OT>& matrix,
                                const DenseVector<IT, NT>& x, DenseVector<IT, NT>& y,
                                cudaStream_t stream = nullptr)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "SpMV semiring value type must match the matrix value type");
  return detail::SpmvCudaBlockPerRowImpl<SemiRing, IT, NT, OT, kBlockSize>::Run(matrix, x, y,
                                                                                stream);
}

}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
