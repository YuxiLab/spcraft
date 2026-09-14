#pragma once

#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/cuCsrMatrix.cuh"
#include "utils/cuda/cuda_error.h"

namespace spcraft
{

template <class IT, class NT, class OT>
cuCsrMatrix<IT, NT, OT>::cuCsrMatrix(const CsrMatrix<IT, NT, OT>& host)
{
  Allocate(host.nnz, host.m, host.n);
  if (host.row_ptr) {
    CUDACHK(cudaMemcpy(row_ptr, host.row_ptr, (static_cast<std::size_t>(host.m) + 1) * sizeof(OT),
                       cudaMemcpyHostToDevice),
            "cuCsrMatrix::FromHost row_ptr");
  }
  if (host.col_id) {
    CUDACHK(cudaMemcpy(col_id, host.col_id, static_cast<std::size_t>(host.nnz) * sizeof(IT),
                       cudaMemcpyHostToDevice),
            "cuCsrMatrix::FromHost col_id");
  }
  if (host.val) {
    CUDACHK(cudaMemcpy(val, host.val, static_cast<std::size_t>(host.nnz) * sizeof(NT),
                       cudaMemcpyHostToDevice),
            "cuCsrMatrix::FromHost val");
  }
}

template <class IT, class NT, class OT>
cuCsrMatrix<IT, NT, OT>::cuCsrMatrix(OT* row_ptr_, IT* col_id_, NT* val_, OT nnz_, IT m_, IT n_)
    : row_ptr(row_ptr_), col_id(col_id_), val(val_), nnz(nnz_), m(m_), n(n_), memowned(false)
{
}

template <class IT, class NT, class OT>
cuCsrMatrix<IT, NT, OT>::cuCsrMatrix(cuCsrMatrix<IT, NT, OT>&& rhs) noexcept
    : row_ptr(rhs.row_ptr),
      col_id(rhs.col_id),
      val(rhs.val),
      nnz(rhs.nnz),
      m(rhs.m),
      n(rhs.n),
      memowned(rhs.memowned)
{
  rhs.Reset();
}

template <class IT, class NT, class OT>
cuCsrMatrix<IT, NT, OT>& cuCsrMatrix<IT, NT, OT>::operator=(cuCsrMatrix<IT, NT, OT>&& rhs) noexcept
{
  if (this != &rhs) {
    SafeDelete(memowned, row_ptr, col_id, val);

    row_ptr = rhs.row_ptr;
    col_id = rhs.col_id;
    val = rhs.val;
    nnz = rhs.nnz;
    m = rhs.m;
    n = rhs.n;
    memowned = rhs.memowned;

    rhs.Reset();
  }
  return *this;
}

template <class IT, class NT, class OT>
cuCsrMatrix<IT, NT, OT>::~cuCsrMatrix()
{
  SafeDelete(memowned, row_ptr, col_id, val);
}

template <class IT, class NT, class OT>
void cuCsrMatrix<IT, NT, OT>::Allocate(OT require_nnz, IT nRows, IT nCols)
{
  if constexpr (std::is_signed_v<IT>) {
    if (nRows < 0 || nCols < 0) {
      throw std::invalid_argument("cuCsrMatrix dimensions must be non-negative");
    }
  }
  if constexpr (std::is_signed_v<OT>) {
    if (require_nnz < 0) {
      throw std::invalid_argument("cuCsrMatrix nnz must be non-negative");
    }
  }

  auto replacement = SafeAllocate(nRows, require_nnz);
  SafeDelete(memowned, row_ptr, col_id, val);
  m = nRows;
  n = nCols;
  nnz = require_nnz;
  memowned = true;
  std::tie(row_ptr, col_id, val) = replacement;
}

template <class IT, class NT, class OT>
cuCsrMatrix<IT, NT, OT> cuCsrMatrix<IT, NT, OT>::Clone() const
{
  cuCsrMatrix<IT, NT, OT> res;
  res.Allocate(nnz, m, n);
  if (row_ptr) {
    CUDACHK(cudaMemcpy(res.row_ptr, row_ptr, (static_cast<std::size_t>(m) + 1) * sizeof(OT),
                       cudaMemcpyDeviceToDevice),
            "cuCsrMatrix::Clone row_ptr");
  }
  if (col_id) {
    CUDACHK(cudaMemcpy(res.col_id, col_id, static_cast<std::size_t>(nnz) * sizeof(IT),
                       cudaMemcpyDeviceToDevice),
            "cuCsrMatrix::Clone col_id");
  }
  if (val) {
    CUDACHK(cudaMemcpy(res.val, val, static_cast<std::size_t>(nnz) * sizeof(NT),
                       cudaMemcpyDeviceToDevice),
            "cuCsrMatrix::Clone val");
  }
  return res;
}

template <class IT, class NT, class OT>
void cuCsrMatrix<IT, NT, OT>::Reset()
{
  row_ptr = nullptr;
  col_id = nullptr;
  val = nullptr;
  nnz = 0;
  m = 0;
  n = 0;
  memowned = false;
}

template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT> cuCsrMatrix<IT, NT, OT>::ToHost() const
{
  CsrMatrix<IT, NT, OT> res;
  res.Allocate(nnz, m, n);
  if (row_ptr) {
    CUDACHK(cudaMemcpy(res.row_ptr, row_ptr, (static_cast<std::size_t>(m) + 1) * sizeof(OT),
                       cudaMemcpyDeviceToHost),
            "cuCsrMatrix::ToHost row_ptr");
  }
  if (col_id) {
    CUDACHK(cudaMemcpy(res.col_id, col_id, static_cast<std::size_t>(nnz) * sizeof(IT),
                       cudaMemcpyDeviceToHost),
            "cuCsrMatrix::ToHost col_id");
  }
  if (val) {
    CUDACHK(cudaMemcpy(res.val, val, static_cast<std::size_t>(nnz) * sizeof(NT),
                       cudaMemcpyDeviceToHost),
            "cuCsrMatrix::ToHost val");
  }
  return res;
}

template <class IT, class NT, class OT>
CsrMatrix<IT, NT, OT> cuCsrMatrix<IT, NT, OT>::View() const
{
  return CsrMatrix<IT, NT, OT>(row_ptr, col_id, val, nnz, m, n);
}

/*************************************
 *             Static Members
 *************************************/

template <class IT, class NT, class OT>
std::tuple<OT*, IT*, NT*> cuCsrMatrix<IT, NT, OT>::SafeAllocate(IT m, OT nnz)
{
  if constexpr (std::is_signed_v<IT>) {
    if (m < 0) throw std::invalid_argument("cuCsrMatrix row count must be non-negative");
  }
  if constexpr (std::is_signed_v<OT>) {
    if (nnz < 0) throw std::invalid_argument("cuCsrMatrix nnz must be non-negative");
  }

  const std::size_t row_bytes = (static_cast<std::size_t>(m) + 1) * sizeof(OT);
  const std::size_t col_bytes = static_cast<std::size_t>(nnz) * sizeof(IT);
  const std::size_t val_bytes = static_cast<std::size_t>(nnz) * sizeof(NT);

  OT* r = nullptr;
  IT* c = nullptr;
  NT* v = nullptr;
  cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&r), row_bytes);
  if (status == cudaSuccess) status = cudaMemset(r, 0, row_bytes);
  if (status == cudaSuccess && nnz != OT{0}) {
    status = cudaMalloc(reinterpret_cast<void**>(&c), col_bytes);
    if (status == cudaSuccess) status = cudaMemset(c, 0, col_bytes);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&v), val_bytes);
    if (status == cudaSuccess) status = cudaMemset(v, 0, val_bytes);
  }
  if (status != cudaSuccess) {
    cudaFree(r);
    cudaFree(c);
    cudaFree(v);
    CUDACHK(status, "cuCsrMatrix allocation");
  }
  return std::make_tuple(r, c, v);
}

template <class IT, class NT, class OT>
void cuCsrMatrix<IT, NT, OT>::SafeDelete(bool memowned, OT* row_ptr, IT* col_id, NT* val)
{
  if (memowned) {
    cudaFree(row_ptr);
    cudaFree(col_id);
    cudaFree(val);
  }
}

}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
