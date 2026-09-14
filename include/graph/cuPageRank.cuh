#pragma once

#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "core/CsrMatrix.h"
#include "core/DenseVector.h"
#include "graph/PageRank.h"

namespace spcraft
{
namespace detail
{

template <typename IT, typename NT, typename OT, int kBlockSize>
struct PageRankCudaImpl {
  static cudaError_t Run(const CsrMatrix<IT, NT, OT>& op, DenseVector<IT, NT>& rank,
                         const PageRankOptions<NT>& options, PageRankResult<NT>* result,
                         cudaStream_t stream);
};

}  // namespace detail

/**
 * @brief PageRank by power iteration on the GPU, driven by the CUDA SpMV kernel.
 *
 * Runs the same recurrence as pagerank_openmp -- one SpMV against the
 * transposed operator, a sum reduction for the dangling mass, then a fused
 * update and L1-residual reduction -- so the two backends agree to rounding.
 *
 * The operator and the rank vector must wrap device memory; build the operator
 * on the host with pagerank_operator() and upload it, for instance through
 * cuCsrMatrix::FromHost(). Scratch space is allocated and freed internally.
 *
 * Convergence is tested on the host, so each iteration ends with one small
 * device-to-host copy and a stream synchronization. The stream is synchronized
 * on return, unlike the SpMV entry point.
 *
 * @param op Transposed operator in device memory; square, n x n.
 * @param rank Output ranks in device memory, size n. Overwritten with 1/n first.
 * @param options Damping, tolerance, and iteration cap.
 * @param result Host-side run summary; may be null.
 * @tparam kBlockSize Power-of-two CUDA block size in the range [32, 1024].
 * @param stream CUDA stream used for every launch.
 * @return cudaErrorInvalidValue for bad shapes or options,
 *         cudaErrorInvalidDevicePointer for null buffers, else the first
 *         failure encountered.
 */
template <typename IT, typename NT, typename OT = IT, int kBlockSize = 256>
cudaError_t PageRankCuda(const CsrMatrix<IT, NT, OT>& op, DenseVector<IT, NT>& rank,
                         const PageRankOptions<NT>& options = PageRankOptions<NT>{},
                         PageRankResult<NT>* result = nullptr, cudaStream_t stream = nullptr)
{
  static_assert(std::is_floating_point_v<NT>, "PageRank requires a floating-point value type");
  return detail::PageRankCudaImpl<IT, NT, OT, kBlockSize>::Run(op, rank, options, result,
                                                               stream);
}

}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
