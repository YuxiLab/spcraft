#include "cuSpMV.cuh"

#include <utility>

namespace spcraft
{
namespace detail
{

template <class IT, class NT, class OT, SpmvBlockSize BLOCK_SIZE>
__global__ __launch_bounds__(static_cast<int>(BLOCK_SIZE)) void spmv_block_per_row_kernel(
    IT num_rows, const OT* row_ptr, const IT* col_id, const NT* values, const NT* x, NT* y)
{
  constexpr int block_size = static_cast<int>(BLOCK_SIZE);
  const IT row = static_cast<IT>(blockIdx.x);
  if (row >= num_rows) return;

  __shared__ NT partial_sums[block_size];

  NT sum = NT{0};
  for (OT entry = row_ptr[row] + static_cast<OT>(threadIdx.x); entry < row_ptr[row + 1];
       entry += static_cast<OT>(block_size)) {
    sum += values[entry] * x[col_id[entry]];
  }
  partial_sums[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned int offset = block_size / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) y[row] = partial_sums[0];
}

template <class IT, class NT, class OT, SpmvBlockSize BLOCK_SIZE>
cudaError_t launch_spmv_block_per_row(const CsrMatrix<IT, NT, OT>& A, const NT* x, NT* y,
                                      cudaStream_t stream)
{
  constexpr int block_size = static_cast<int>(BLOCK_SIZE);
  static_assert(block_size >= 1 && block_size <= 1024,
                "BLOCK_SIZE must be in the range [1, 1024]");
  static_assert((block_size & (block_size - 1)) == 0, "BLOCK_SIZE must be a power of two");

  if (A.m == 0) return cudaSuccess;
  if (A.row_ptr == nullptr || A.col_id == nullptr || A.val == nullptr || x == nullptr ||
      y == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }

  spmv_block_per_row_kernel<IT, NT, OT, BLOCK_SIZE>
      <<<A.m, block_size, 0, stream>>>(A.m, A.row_ptr, A.col_id, A.val, x, y);
  return cudaGetLastError();
}

template <class Callback, auto... OPTIONS>
struct Selection {
  Callback& callback;
};

struct EnumFinish {
  template <class Callback, auto... OPTIONS>
  static cudaError_t call(Selection<Callback, OPTIONS...> selection)
  {
    return selection.callback.template call<OPTIONS...>();
  }
};

template <class Next, auto CURRENT, auto... REST>
struct EnumOption {
  template <class Callback, auto... OPTIONS, class... Values>
  static cudaError_t call(Selection<Callback, OPTIONS...> selection, int value,
                          Values... values)
  {
    if (value == static_cast<int>(CURRENT)) {
      return Next::call(Selection<Callback, OPTIONS..., CURRENT>{selection.callback},
                        values...);
    }
    if constexpr (sizeof...(REST) > 0) {
      return EnumOption<Next, REST...>::call(selection, value, values...);
    }
    return cudaErrorInvalidValue;
  }
};

using BlockSizeOptions =
    EnumOption<EnumFinish, SpmvBlockSize::B32, SpmvBlockSize::B64, SpmvBlockSize::B128,
               SpmvBlockSize::B256, SpmvBlockSize::B512, SpmvBlockSize::B1024>;
using NumTypeOptions =
    EnumOption<BlockSizeOptions, NumTypeEnum::Float32, NumTypeEnum::Float64>;
using OffsetTypeOptions =
    EnumOption<NumTypeOptions, OffsetTypeEnum::Int32, OffsetTypeEnum::Int64>;
using IndexTypeOptions =
    EnumOption<OffsetTypeOptions, IndexTypeEnum::Int32, IndexTypeEnum::Int64>;

struct SpmvCall {
  std::int64_t rows;
  std::int64_t columns;
  std::int64_t nonzeros;
  void* row_ptr;
  void* col_id;
  void* values;
  const void* x;
  void* y;
  cudaStream_t stream;

  template <IndexTypeEnum INDEX_TYPE, OffsetTypeEnum OFFSET_TYPE, NumTypeEnum NUM_TYPE,
            SpmvBlockSize BLOCK_SIZE>
  cudaError_t call()
  {
    using IT = index_type_t<INDEX_TYPE>;
    using OT = offset_type_t<OFFSET_TYPE>;
    using NT = num_type_t<NUM_TYPE>;

    CsrMatrix<IT, NT, OT> matrix(static_cast<OT*>(row_ptr), static_cast<IT*>(col_id),
                                 static_cast<NT*>(values), static_cast<OT>(nonzeros),
                                 static_cast<IT>(rows), static_cast<IT>(columns));
    return launch_spmv_block_per_row<IT, NT, OT, BLOCK_SIZE>(matrix, static_cast<const NT*>(x),
                                                             static_cast<NT*>(y), stream);
  }
};

cudaError_t spmv_cuda_block_per_row_dispatch(IndexTypeEnum index_type,
                                             OffsetTypeEnum offset_type, NumTypeEnum num_type,
                                             SpmvBlockSize block_size, std::int64_t rows,
                                             std::int64_t columns, std::int64_t nonzeros,
                                             void* row_ptr, void* col_id, void* values,
                                             const void* x, void* y, cudaStream_t stream)
{
  if (rows < 0 || columns < 0 || nonzeros < 0) return cudaErrorInvalidValue;

  SpmvCall call{rows, columns, nonzeros, row_ptr, col_id, values, x, y, stream};
  return IndexTypeOptions::call(Selection<SpmvCall>{call}, static_cast<int>(index_type),
                                static_cast<int>(offset_type), static_cast<int>(num_type),
                                static_cast<int>(block_size));
}

}  // namespace detail
}  // namespace spcraft
