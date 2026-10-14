#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "core/AllocationGuard.h"
#include "core/cuCooMatrix.cuh"
#include "utils/cuda/cuda_error.h"

namespace spcraft
{

SP_MAT_TEMP
cuCooMatrix<IT, NT, OT>::cuCooMatrix() noexcept = default;

SP_MAT_TEMP
cuCooMatrix<IT, NT, OT>::cuCooMatrix(Entry* entries_, OT count, IT rows, IT columns)
    : entries(entries_), nnz(count), m(rows), n(columns), memowned(false)
{
}

SP_MAT_TEMP
cuCooMatrix<IT, NT, OT>::cuCooMatrix(const SPCOO& host)
{
  // A local owner also frees device storage when an upload throws.
  cuCooMatrix replacement;
  replacement.Allocate(host.nnz, host.m, host.n);
  if (host.nnz != OT{0}) {
    if (!host.entries) throw std::invalid_argument("cuCooMatrix upload requires host entries");
    const auto bytes = static_cast<std::size_t>(host.nnz) * sizeof(Entry);
    CUDACHK(cudaMemcpy(replacement.entries, host.entries, bytes, cudaMemcpyHostToDevice),
            "cuCooMatrix::FromHost entries");
  }
  *this = std::move(replacement);
}

SP_MAT_TEMP
cuCooMatrix<IT, NT, OT>::cuCooMatrix(cuCooMatrix&& rhs) noexcept
    : entries(rhs.entries), nnz(rhs.nnz), m(rhs.m), n(rhs.n), memowned(rhs.memowned)
{
  rhs.Reset();
}

SP_MAT_TEMP
cuCooMatrix<IT, NT, OT>& cuCooMatrix<IT, NT, OT>::operator=(cuCooMatrix&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, entries);
    entries = rhs.entries;
    nnz = rhs.nnz;
    m = rhs.m;
    n = rhs.n;
    memowned = rhs.memowned;
    rhs.Reset();
  }
  return *this;
}

SP_MAT_TEMP
cuCooMatrix<IT, NT, OT>::~cuCooMatrix() { SafeDelete(memowned, entries); }

SP_MAT_TEMP
void cuCooMatrix<IT, NT, OT>::Allocate(OT count, IT rows, IT columns)
{
  Entry* replacement = SafeAllocate(count, rows, columns);
  SafeDelete(memowned, entries);
  entries = replacement;
  nnz = count;
  m = rows;
  n = columns;
  memowned = true;
}

SP_MAT_TEMP
cuCooMatrix<IT, NT, OT> cuCooMatrix<IT, NT, OT>::Clone() const
{
  cuCooMatrix result;
  result.Allocate(nnz, m, n);
  if (nnz != OT{0}) {
    const auto bytes = static_cast<std::size_t>(nnz) * sizeof(Entry);
    CUDACHK(cudaMemcpy(result.entries, entries, bytes, cudaMemcpyDeviceToDevice),
            "cuCooMatrix::Clone entries");
  }
  return result;
}

SP_MAT_TEMP
void cuCooMatrix<IT, NT, OT>::Reset()
{
  entries = nullptr;
  nnz = 0;
  m = 0;
  n = 0;
  memowned = false;
}

SP_MAT_TEMP
SPCOO cuCooMatrix<IT, NT, OT>::ToHost() const
{
  SPCOO result;
  result.Allocate(nnz, m, n);
  if (nnz != OT{0}) {
    const auto bytes = static_cast<std::size_t>(nnz) * sizeof(Entry);
    CUDACHK(cudaMemcpy(result.entries, entries, bytes, cudaMemcpyDeviceToHost),
            "cuCooMatrix::ToHost entries");
  }
  return result;
}

SP_MAT_TEMP
SPCOO cuCooMatrix<IT, NT, OT>::View() const { return SPCOO(entries, nnz, m, n); }

SP_MAT_TEMP
typename cuCooMatrix<IT, NT, OT>::Entry* cuCooMatrix<IT, NT, OT>::SafeAllocate(OT count, IT rows,
                                                                               IT columns)
{
  static_assert(std::is_trivially_copyable_v<Entry>);
  if constexpr (std::is_signed_v<IT>) {
    if (rows < 0 || columns < 0) {
      throw std::invalid_argument("cuCooMatrix dimensions must be non-negative");
    }
  }
  if constexpr (std::is_signed_v<OT>) {
    if (count < 0) throw std::invalid_argument("cuCooMatrix nnz must be non-negative");
  }
  if ((rows == 0 || columns == 0) && count != OT{0}) {
    throw std::invalid_argument("nonempty cuCooMatrix requires nonzero dimensions");
  }
  const auto entry_count = detail::CheckedElementCount<Entry>(count);
  if (entry_count == 0) return nullptr;
  const auto bytes = entry_count * sizeof(Entry);
  Entry* replacement = nullptr;
  cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&replacement), bytes);
  if (status == cudaSuccess) status = cudaMemset(replacement, 0, bytes);
  if (status != cudaSuccess) {
    cudaFree(replacement);
    CUDACHK(status, "cuCooMatrix allocation");
  }
  return replacement;
}

SP_MAT_TEMP
void cuCooMatrix<IT, NT, OT>::SafeDelete(bool owned, Entry* entries)
{
  if (owned && entries) cudaFree(entries);
}

#define SPCRAFT_DEFINE_CUCOO_MATRIX(r, product)                                   \
  template class cuCooMatrix<BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                             BOOST_PP_SEQ_ELEM(1, product),                       \
                             BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product))>;

// Match the supported host storage pairs, including Boolean graph storage.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_CUCOO_MATRIX,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)(bool)))
// clang-format on
#undef SPCRAFT_DEFINE_CUCOO_MATRIX

}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
