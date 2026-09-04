#include "kernel/cuSpMV.cuh"

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

namespace spcraft
{

namespace detail
{
template <class SemiRing, typename IT, typename NT, typename OT, int kBlockSize>
__global__ __launch_bounds__(kBlockSize) void SpmvBlockPerRowKernel(
    IT num_rows, const OT* row_ptr, const IT* col_id, const NT* values, const NT* x, NT* y)
{
  constexpr int block_size = kBlockSize;
  const IT row = static_cast<IT>(blockIdx.x);
  if (row >= num_rows) return;

  __shared__ NT partial_sums[block_size];

  NT sum = SemiRing::kAdditiveIdentity;
  for (OT entry = row_ptr[row] + static_cast<OT>(threadIdx.x); entry < row_ptr[row + 1];
       entry += static_cast<OT>(block_size)) {
    sum = SemiRing::Add(sum, SemiRing::Multiply(values[entry], x[col_id[entry]]));
  }
  partial_sums[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned int offset = block_size / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      partial_sums[threadIdx.x] =
          SemiRing::Add(partial_sums[threadIdx.x], partial_sums[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) y[row] = partial_sums[0];
}

template <class SemiRing, typename IT, typename NT, typename OT, int kBlockSize>
cudaError_t SpmvCudaBlockPerRowImpl<SemiRing, IT, NT, OT, kBlockSize>::Run(
    const CsrMatrix<IT, NT, OT>& matrix, const DenseVector<IT, NT>& x, DenseVector<IT, NT>& y,
    cudaStream_t stream)
{
  constexpr int block_size = kBlockSize;
  static_assert(block_size >= 1 && block_size <= 1024,
                "BLOCK_SIZE must be in the range [1, 1024]");
  static_assert((block_size & (block_size - 1)) == 0, "BLOCK_SIZE must be a power of two");

  if (matrix.m < IT{0} || matrix.n < IT{0} || matrix.nnz < OT{0}) {
    return cudaErrorInvalidValue;
  }
  if (x.n != matrix.n || y.n != matrix.m) return cudaErrorInvalidValue;
  if (matrix.m == 0) return cudaSuccess;
  if (matrix.row_ptr == nullptr || matrix.col_id == nullptr || matrix.val == nullptr ||
      x.val == nullptr || y.val == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }

  SpmvBlockPerRowKernel<SemiRing, IT, NT, OT, kBlockSize><<<matrix.m, block_size, 0, stream>>>(
      matrix.m, matrix.row_ptr, matrix.col_id, matrix.val, x.val, y.val);
  return cudaGetLastError();
}

#define SPCRAFT_DEFINE_SPMV_CUDA_BLOCK_PER_ROW(r, product)                              \
  template struct SpmvCudaBlockPerRowImpl<PlusTimesRing<BOOST_PP_SEQ_ELEM(1, product)>, \
                                          BOOST_PP_SEQ_ENUM(product)>;
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_SPMV_CUDA_BLOCK_PER_ROW,
                              ((std::int32_t)(std::int64_t))
                              ((float)(double)(bool))
                              ((std::int32_t)(std::int64_t))
                              ((32)(64)(128)(256)(512)(1024)))
// clang-format on
#undef SPCRAFT_DEFINE_SPMV_CUDA_BLOCK_PER_ROW

}  // namespace detail
}  // namespace spcraft
