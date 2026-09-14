#include "graph/cuPageRank.cuh"

#include <cstddef>

#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "kernel/cuSpMV.cuh"
#include "semiring/SemiRing.h"

namespace spcraft
{

namespace detail
{

//! Tree reduction across one block. Correct only for power-of-two block sizes.
template <typename NT, int kBlockSize>
__device__ NT PageRankBlockSum(NT value)
{
  __shared__ NT partial_sums[kBlockSize];

  partial_sums[threadIdx.x] = value;
  __syncthreads();

  for (unsigned int offset = kBlockSize / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset];
    }
    __syncthreads();
  }

  return partial_sums[0];
}

template <typename IT, typename NT, int kBlockSize>
__global__ __launch_bounds__(kBlockSize) void PageRankFillKernel(NT* values, IT n, NT value)
{
  const IT stride = static_cast<IT>(gridDim.x) * static_cast<IT>(kBlockSize);
  for (IT index =
           static_cast<IT>(blockIdx.x) * static_cast<IT>(kBlockSize) + static_cast<IT>(threadIdx.x);
       index < n; index += stride) {
    values[index] = value;
  }
}

//! One partial sum per block, so a second launch with a single block finishes it.
template <typename IT, typename NT, int kBlockSize>
__global__ __launch_bounds__(kBlockSize) void PageRankSumKernel(const NT* input, IT n, NT* partials)
{
  const IT stride = static_cast<IT>(gridDim.x) * static_cast<IT>(kBlockSize);
  NT sum = NT{0};
  for (IT index =
           static_cast<IT>(blockIdx.x) * static_cast<IT>(kBlockSize) + static_cast<IT>(threadIdx.x);
       index < n; index += stride) {
    sum += input[index];
  }

  sum = PageRankBlockSum<NT, kBlockSize>(sum);
  if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

/**
 * @brief Apply the damping update in place and reduce the L1 change.
 *
 * `total` points at sum(y) in device memory, so the dangling mass is folded in
 * without a round trip to the host.
 */
template <typename IT, typename NT, int kBlockSize>
__global__ __launch_bounds__(kBlockSize) void PageRankUpdateKernel(IT n, const NT* y, NT* rank,
                                                                   const NT* total, NT damping,
                                                                   NT uniform, NT* partials)
{
  const NT base = ((NT{1} - damping) + damping * (NT{1} - *total)) * uniform;

  const IT stride = static_cast<IT>(gridDim.x) * static_cast<IT>(kBlockSize);
  NT residual = NT{0};
  for (IT index =
           static_cast<IT>(blockIdx.x) * static_cast<IT>(kBlockSize) + static_cast<IT>(threadIdx.x);
       index < n; index += stride) {
    const NT updated = base + damping * y[index];
    const NT difference = updated - rank[index];
    residual += difference < NT{0} ? -difference : difference;
    rank[index] = updated;
  }

  residual = PageRankBlockSum<NT, kBlockSize>(residual);
  if (threadIdx.x == 0) partials[blockIdx.x] = residual;
}

template <typename IT, typename NT, typename OT, int kBlockSize>
cudaError_t PageRankCudaImpl<IT, NT, OT, kBlockSize>::Run(const CsrMatrix<IT, NT, OT>& op,
                                                          DenseVector<IT, NT>& rank,
                                                          const PageRankOptions<NT>& options,
                                                          PageRankResult<NT>* result,
                                                          cudaStream_t stream)
{
  constexpr int block_size = kBlockSize;
  static_assert(block_size >= 32 && block_size <= 1024,
                "BLOCK_SIZE must be in the range [32, 1024]");
  static_assert((block_size & (block_size - 1)) == 0, "BLOCK_SIZE must be a power of two");

  if (result != nullptr) *result = PageRankResult<NT>{};

  if constexpr (std::is_signed_v<IT>) {
    if (op.m < IT{0} || op.n < IT{0}) return cudaErrorInvalidValue;
  }
  if constexpr (std::is_signed_v<OT>) {
    if (op.nnz < OT{0}) return cudaErrorInvalidValue;
  }
  if (op.m != op.n || rank.n != op.m) return cudaErrorInvalidValue;
  if (!PageRankOptionsAreValid(options)) return cudaErrorInvalidValue;

  const IT n = op.m;
  if (n == IT{0}) {
    if (result != nullptr) result->converged = true;
    return cudaSuccess;
  }

  if (rank.val == nullptr || op.row_ptr == nullptr) return cudaErrorInvalidDevicePointer;
  if (op.nnz != OT{0} && (op.col_id == nullptr || op.val == nullptr)) {
    return cudaErrorInvalidDevicePointer;
  }

  // Cap the grid so the per-block partials always fit in one block, and the
  // second reduction stage is a single launch rather than a loop.
  const IT chunks = (n + static_cast<IT>(block_size) - 1) / static_cast<IT>(block_size);
  const unsigned int grid = chunks > static_cast<IT>(block_size)
                                ? static_cast<unsigned int>(block_size)
                                : static_cast<unsigned int>(chunks);

  // One allocation for the SpMV output, the block partials, and the two scalars.
  NT* workspace = nullptr;
  const std::size_t workspace_count = static_cast<std::size_t>(n) + grid + 2;
  cudaError_t status =
      cudaMalloc(reinterpret_cast<void**>(&workspace), workspace_count * sizeof(NT));
  if (status != cudaSuccess) return status;

  NT* y = workspace;
  NT* partials = workspace + static_cast<std::size_t>(n);
  NT* total = partials + grid;
  NT* residual = total + 1;

  const NT damping = options.damping;
  const NT uniform = NT{1} / static_cast<NT>(n);

  PageRankFillKernel<IT, NT, kBlockSize><<<grid, block_size, 0, stream>>>(rank.val, n, uniform);
  status = cudaGetLastError();

  // Non-owning view over the scratch buffer, so the SpMV kernel can write to it
  // without DenseVector taking ownership of device memory it must not free.
  DenseVector<IT, NT> y_output(y, n);

  for (int iteration = 1; status == cudaSuccess && iteration <= options.max_iterations;
       ++iteration) {
    if (op.nnz != OT{0}) {
      // The one place the algorithm touches the graph: y = P^T . rank.
      status = SpmvCudaBlockPerRow<PlusTimesRing<NT>, IT, NT, OT, kBlockSize>(op, rank, y_output,
                                                                              stream);
    } else {
      // No edges survived normalization; every vertex is dangling.
      PageRankFillKernel<IT, NT, kBlockSize><<<grid, block_size, 0, stream>>>(y, n, NT{0});
      status = cudaGetLastError();
    }
    if (status != cudaSuccess) break;

    PageRankSumKernel<IT, NT, kBlockSize><<<grid, block_size, 0, stream>>>(y, n, partials);
    PageRankSumKernel<IT, NT, kBlockSize>
        <<<1, block_size, 0, stream>>>(partials, static_cast<IT>(grid), total);
    PageRankUpdateKernel<IT, NT, kBlockSize>
        <<<grid, block_size, 0, stream>>>(n, y, rank.val, total, damping, uniform, partials);
    PageRankSumKernel<IT, NT, kBlockSize>
        <<<1, block_size, 0, stream>>>(partials, static_cast<IT>(grid), residual);
    status = cudaGetLastError();
    if (status != cudaSuccess) break;

    NT host_residual = NT{0};
    status = cudaMemcpyAsync(&host_residual, residual, sizeof(NT), cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) break;
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) break;

    if (result != nullptr) {
      result->iterations = iteration;
      result->residual = host_residual;
    }
    if (host_residual <= options.tolerance) {
      if (result != nullptr) result->converged = true;
      break;
    }
  }

  const cudaError_t free_status = cudaFree(workspace);
  return status != cudaSuccess ? status : free_status;
}

#define SPCRAFT_DEFINE_PAGERANK_CUDA(r, product) \
  template struct PageRankCudaImpl<BOOST_PP_SEQ_ENUM(product)>;
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_PAGERANK_CUDA,
                              ((std::int32_t)(std::int64_t))
                              ((float)(double))
                              ((std::int32_t)(std::int64_t))
                              ((32)(64)(128)(256)(512)(1024)))
// clang-format on
#undef SPCRAFT_DEFINE_PAGERANK_CUDA

}  // namespace detail
}  // namespace spcraft
