#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include <cstddef>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/AllocationGuard.h"
#include "core/cuCscMatrix.cuh"
#include "utils/cuda/cuda_error.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

SP_MAT_TEMP
cuCscMatrix<IT, NT, OT>::cuCscMatrix() noexcept = default;

SP_MAT_TEMP
cuCscMatrix<IT, NT, OT>::cuCscMatrix(const SPCSC& host)
{
  // A constructor's destructor does not run if a transfer throws.
  cuCscMatrix replacement;
  replacement.Allocate(host.nnz, host.m, host.n);
  if (host.nnz != OT{0} && (!host.col_ptr || !host.row_id || !host.val)) {
    throw std::invalid_argument("cuCscMatrix upload requires nonnull host buffers");
  }
  if (host.col_ptr) {
    const auto bytes = (static_cast<std::size_t>(host.n) + 1) * sizeof(OT);
    CUDACHK(cudaMemcpy(replacement.col_ptr, host.col_ptr, bytes, cudaMemcpyHostToDevice),
            "cuCscMatrix::FromHost col_ptr");
  }
  if (host.nnz != OT{0}) {
    const auto index_bytes = static_cast<std::size_t>(host.nnz) * sizeof(IT);
    const auto value_bytes = static_cast<std::size_t>(host.nnz) * sizeof(NT);
    CUDACHK(cudaMemcpy(replacement.row_id, host.row_id, index_bytes, cudaMemcpyHostToDevice),
            "cuCscMatrix::FromHost row_id");
    CUDACHK(cudaMemcpy(replacement.val, host.val, value_bytes, cudaMemcpyHostToDevice),
            "cuCscMatrix::FromHost val");
  }
  *this = std::move(replacement);
}

SP_MAT_TEMP
cuCscMatrix<IT, NT, OT>::cuCscMatrix(OT* col_ptr_, IT* row_id_, NT* val_, OT nnz_, IT m_, IT n_)
    : col_ptr(col_ptr_), row_id(row_id_), val(val_), nnz(nnz_), m(m_), n(n_), memowned(false)
{
}

SP_MAT_TEMP
cuCscMatrix<IT, NT, OT>::cuCscMatrix(cuCscMatrix<IT, NT, OT>&& rhs) noexcept
    : col_ptr(rhs.col_ptr),
      row_id(rhs.row_id),
      val(rhs.val),
      nnz(rhs.nnz),
      m(rhs.m),
      n(rhs.n),
      memowned(rhs.memowned)
{
  rhs.Reset();
}

SP_MAT_TEMP
cuCscMatrix<IT, NT, OT>& cuCscMatrix<IT, NT, OT>::operator=(cuCscMatrix<IT, NT, OT>&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, col_ptr, row_id, val);

    col_ptr = rhs.col_ptr;
    row_id = rhs.row_id;
    val = rhs.val;
    nnz = rhs.nnz;
    m = rhs.m;
    n = rhs.n;
    memowned = rhs.memowned;

    rhs.Reset();
  }
  return *this;
}

SP_MAT_TEMP
cuCscMatrix<IT, NT, OT>::~cuCscMatrix() { SafeDelete(memowned, col_ptr, row_id, val); }

SP_MAT_TEMP
void cuCscMatrix<IT, NT, OT>::Allocate(OT require_nnz, IT nRows, IT nCols)
{
  if constexpr (std::is_signed_v<IT>) {
    if (nRows < 0 || nCols < 0) {
      throw std::invalid_argument("cuCscMatrix dimensions must be non-negative");
    }
  }
  if constexpr (std::is_signed_v<OT>) {
    if (require_nnz < 0) {
      throw std::invalid_argument("cuCscMatrix nnz must be non-negative");
    }
  }

  if ((nRows == 0 || nCols == 0) && require_nnz != OT{0}) {
    throw std::invalid_argument("nonempty cuCscMatrix requires nonzero dimensions");
  }
  auto replacement = SafeAllocate(nCols, require_nnz);
  SafeDelete(memowned, col_ptr, row_id, val);
  m = nRows;
  n = nCols;
  nnz = require_nnz;
  memowned = true;
  std::tie(col_ptr, row_id, val) = replacement;
}

SP_MAT_TEMP
cuCscMatrix<IT, NT, OT> cuCscMatrix<IT, NT, OT>::Clone() const
{
  cuCscMatrix<IT, NT, OT> res;
  res.Allocate(nnz, m, n);
  if (col_ptr) {
    CUDACHK(cudaMemcpy(res.col_ptr, col_ptr, (static_cast<std::size_t>(n) + 1) * sizeof(OT),
                       cudaMemcpyDeviceToDevice),
            "cuCscMatrix::Clone col_ptr");
  }
  if (row_id) {
    CUDACHK(cudaMemcpy(res.row_id, row_id, static_cast<std::size_t>(nnz) * sizeof(IT),
                       cudaMemcpyDeviceToDevice),
            "cuCscMatrix::Clone row_id");
  }
  if (val) {
    CUDACHK(cudaMemcpy(res.val, val, static_cast<std::size_t>(nnz) * sizeof(NT),
                       cudaMemcpyDeviceToDevice),
            "cuCscMatrix::Clone val");
  }
  return res;
}

SP_MAT_TEMP
void cuCscMatrix<IT, NT, OT>::Reset()
{
  col_ptr = nullptr;
  row_id = nullptr;
  val = nullptr;
  nnz = 0;
  m = 0;
  n = 0;
  memowned = false;
}

SP_MAT_TEMP
SPCSC cuCscMatrix<IT, NT, OT>::ToHost() const
{
  SPCSC res;
  res.Allocate(nnz, m, n);
  if (!col_ptr) res.col_ptr[0] = OT{0};
  if (col_ptr) {
    CUDACHK(cudaMemcpy(res.col_ptr, col_ptr, (static_cast<std::size_t>(n) + 1) * sizeof(OT),
                       cudaMemcpyDeviceToHost),
            "cuCscMatrix::ToHost col_ptr");
  }
  if (row_id) {
    CUDACHK(cudaMemcpy(res.row_id, row_id, static_cast<std::size_t>(nnz) * sizeof(IT),
                       cudaMemcpyDeviceToHost),
            "cuCscMatrix::ToHost row_id");
  }
  if (val) {
    CUDACHK(cudaMemcpy(res.val, val, static_cast<std::size_t>(nnz) * sizeof(NT),
                       cudaMemcpyDeviceToHost),
            "cuCscMatrix::ToHost val");
  }
  return res;
}

SP_MAT_TEMP
SPCSC cuCscMatrix<IT, NT, OT>::View() const { return SPCSC(col_ptr, row_id, val, nnz, m, n); }

/*************************************
 *             Static Members
 *************************************/

SP_MAT_TEMP
std::tuple<OT*, IT*, NT*> cuCscMatrix<IT, NT, OT>::SafeAllocate(IT n, OT nnz)
{
  if constexpr (std::is_signed_v<IT>) {
    if (n < 0) throw std::invalid_argument("cuCscMatrix column count must be non-negative");
  }
  if constexpr (std::is_signed_v<OT>) {
    if (nnz < 0) throw std::invalid_argument("cuCscMatrix nnz must be non-negative");
  }

  // Size every buffer before acquiring any device storage.
  const auto offsets = detail::CheckedElementCount<OT>(n);
  const auto pointer_count = detail::CheckedElementCount<OT>(offsets + 1);
  const auto index_count = detail::CheckedElementCount<IT>(nnz);
  const auto value_count = detail::CheckedElementCount<NT>(nnz);
  const auto pointer_bytes = pointer_count * sizeof(OT);
  const auto index_bytes = index_count * sizeof(IT);
  const auto value_bytes = value_count * sizeof(NT);

  OT* r = nullptr;
  IT* c = nullptr;
  NT* v = nullptr;
  cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&r), pointer_bytes);
  if (status == cudaSuccess) status = cudaMemset(r, 0, pointer_bytes);
  if (status == cudaSuccess && nnz != OT{0}) {
    status = cudaMalloc(reinterpret_cast<void**>(&c), index_bytes);
    if (status == cudaSuccess) status = cudaMemset(c, 0, index_bytes);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&v), value_bytes);
    if (status == cudaSuccess) status = cudaMemset(v, 0, value_bytes);
  }
  if (status != cudaSuccess) {
    cudaFree(r);
    cudaFree(c);
    cudaFree(v);
    CUDACHK(status, "cuCscMatrix allocation");
  }
  return std::make_tuple(r, c, v);
}

SP_MAT_TEMP
void cuCscMatrix<IT, NT, OT>::SafeDelete(bool memowned, OT* col_ptr, IT* row_id, NT* val)
{
  if (memowned) {
    cudaFree(col_ptr);
    cudaFree(row_id);
    cudaFree(val);
  }
}

#define SPCRAFT_DEFINE_CUCSC_MATRIX(r, product)                                   \
  template class cuCscMatrix<BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                             BOOST_PP_SEQ_ELEM(1, product),                       \
                             BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product))>;

// Match the supported host storage pairs, including Boolean graph storage.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_CUCSC_MATRIX,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)(bool)))
// clang-format on
#undef SPCRAFT_DEFINE_CUCSC_MATRIX

}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
