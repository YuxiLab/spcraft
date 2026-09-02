#pragma once

#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "CsrMatrix.h"

namespace spcraft
{

enum class IndexTypeEnum
{
  Int32,
  Int64
};

enum class OffsetTypeEnum
{
  Int32,
  Int64
};

enum class NumTypeEnum
{
  Float32,
  Float64
};

template <IndexTypeEnum>
struct IndexTypeMap;

template <>
struct IndexTypeMap<IndexTypeEnum::Int32>
{
  using type = std::int32_t;
};

template <>
struct IndexTypeMap<IndexTypeEnum::Int64>
{
  using type = std::int64_t;
};

template <OffsetTypeEnum>
struct OffsetTypeMap;

template <>
struct OffsetTypeMap<OffsetTypeEnum::Int32>
{
  using type = std::int32_t;
};

template <>
struct OffsetTypeMap<OffsetTypeEnum::Int64>
{
  using type = std::int64_t;
};

template <NumTypeEnum>
struct NumTypeMap;

template <>
struct NumTypeMap<NumTypeEnum::Float32>
{
  using type = float;
};

template <>
struct NumTypeMap<NumTypeEnum::Float64>
{
  using type = double;
};

template <IndexTypeEnum Value>
using index_type_t = typename IndexTypeMap<Value>::type;

template <OffsetTypeEnum Value>
using offset_type_t = typename OffsetTypeMap<Value>::type;

template <NumTypeEnum Value>
using num_type_t = typename NumTypeMap<Value>::type;

template <class>
struct IndexTypeEnumValue;

template <>
struct IndexTypeEnumValue<std::int32_t>
{
  static constexpr IndexTypeEnum value = IndexTypeEnum::Int32;
};

template <>
struct IndexTypeEnumValue<std::int64_t>
{
  static constexpr IndexTypeEnum value = IndexTypeEnum::Int64;
};

template <class>
struct OffsetTypeEnumValue;

template <>
struct OffsetTypeEnumValue<std::int32_t>
{
  static constexpr OffsetTypeEnum value = OffsetTypeEnum::Int32;
};

template <>
struct OffsetTypeEnumValue<std::int64_t>
{
  static constexpr OffsetTypeEnum value = OffsetTypeEnum::Int64;
};

template <class>
struct NumTypeEnumValue;

template <>
struct NumTypeEnumValue<float>
{
  static constexpr NumTypeEnum value = NumTypeEnum::Float32;
};

template <>
struct NumTypeEnumValue<double>
{
  static constexpr NumTypeEnum value = NumTypeEnum::Float64;
};

enum class SpmvBlockSize : int
{
  B32 = 32,
  B64 = 64,
  B128 = 128,
  B256 = 256,
  B512 = 512,
  B1024 = 1024
};

namespace detail
{

cudaError_t spmv_cuda_block_per_row_dispatch(IndexTypeEnum index_type,
                                              OffsetTypeEnum offset_type,
                                              NumTypeEnum num_type,
                                              SpmvBlockSize block_size,
                                              std::int64_t rows,
                                              std::int64_t columns,
                                              std::int64_t nonzeros,
                                              void* row_ptr,
                                              void* col_id,
                                              void* values,
                                              const void* x,
                                              void* y,
                                              cudaStream_t stream);

}  // namespace detail

/**
 * @brief Compute y = A * x for a CSR matrix using one CUDA block per row.
 *
 * IT is the column-index and row-count type, NT is the numerical type, and OT
 * is the row-offset type. The CSR matrix must wrap device-memory buffers.
 * The launch is asynchronous with respect to the host.
 *
 * @param A CSR matrix view whose buffers reside in device memory.
 * @param x Dense input vector.
 * @param y Dense output vector.
 * @tparam BLOCK_SIZE Power-of-two CUDA block size in the range [1, 1024].
 * @param stream CUDA stream used for the kernel launch.
 */
template <class IT, class NT, class OT = IT, SpmvBlockSize BLOCK_SIZE = SpmvBlockSize::B256>
cudaError_t spmv_cuda_block_per_row(const CsrMatrix<IT, NT, OT>& A,
                                    const NT* x,
                                    NT* y,
                                    cudaStream_t stream = nullptr)
{
  return detail::spmv_cuda_block_per_row_dispatch(
      IndexTypeEnumValue<IT>::value,
      OffsetTypeEnumValue<OT>::value,
      NumTypeEnumValue<NT>::value,
      BLOCK_SIZE,
      static_cast<std::int64_t>(A.m),
      static_cast<std::int64_t>(A.n),
      static_cast<std::int64_t>(A.nnz),
      A.row_ptr,
      A.col_id,
      A.val,
      x,
      y,
      stream);
}

}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
