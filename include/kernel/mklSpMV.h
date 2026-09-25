#pragma once

#ifdef SPCRAFT_USE_MKL

#include <mkl.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/CsrMatrix.h"

namespace spcraft
{
namespace detail
{

inline void CheckMklSparseStatus(sparse_status_t status, const char* operation)
{
  if (status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with oneMKL sparse status " +
                             std::to_string(static_cast<int>(status)));
  }
}

template <class NT>
sparse_status_t CreateMklCsr(sparse_matrix_t*, MKL_INT, MKL_INT, MKL_INT*, MKL_INT*, MKL_INT*, NT*);

template <>
inline sparse_status_t CreateMklCsr<float>(sparse_matrix_t* handle, MKL_INT rows, MKL_INT columns,
                                           MKL_INT* row_start, MKL_INT* row_end,
                                           MKL_INT* column_ids, float* values)
{
  return mkl_sparse_s_create_csr(handle, SPARSE_INDEX_BASE_ZERO, rows, columns, row_start, row_end,
                                 column_ids, values);
}

template <>
inline sparse_status_t CreateMklCsr<double>(sparse_matrix_t* handle, MKL_INT rows, MKL_INT columns,
                                            MKL_INT* row_start, MKL_INT* row_end,
                                            MKL_INT* column_ids, double* values)
{
  return mkl_sparse_d_create_csr(handle, SPARSE_INDEX_BASE_ZERO, rows, columns, row_start, row_end,
                                 column_ids, values);
}

template <class NT>
sparse_status_t MklSparseMv(sparse_matrix_t, matrix_descr, const NT*, NT*);

template <>
inline sparse_status_t MklSparseMv<float>(sparse_matrix_t handle, matrix_descr descriptor,
                                          const float* x, float* y)
{
  return mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0F, handle, descriptor, x, 0.0F, y);
}

template <>
inline sparse_status_t MklSparseMv<double>(sparse_matrix_t handle, matrix_descr descriptor,
                                           const double* x, double* y)
{
  return mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, handle, descriptor, x, 0.0, y);
}

}  // namespace detail

/**
 * @brief Reusable oneMKL inspector-executor handle for a CSR matrix.
 *
 * The matrix owns the arrays; this object only borrows them. Keep the matrix alive
 * for the lifetime of this handle. Handle construction and optimization are intended
 * to happen outside timed SpMV loops.
 */
template <class IT, class NT, class OT = IT>
class MklCsrSpmv
{
  static_assert(std::is_same_v<IT, MKL_INT>, "MklCsrSpmv requires column indices matching MKL_INT");
  static_assert(std::is_same_v<OT, MKL_INT>, "MklCsrSpmv requires row offsets matching MKL_INT");
  static_assert(std::is_same_v<NT, float> || std::is_same_v<NT, double>,
                "MklCsrSpmv supports float and double values");

 public:
  explicit MklCsrSpmv(const CsrMatrix<IT, NT, OT>& matrix, MKL_INT expected_calls = 1000)
  {
    detail::CheckMklSparseStatus(
        detail::CreateMklCsr<NT>(&handle_, matrix.m, matrix.n, matrix.row_ptr, matrix.row_ptr + 1,
                                 matrix.col_id, matrix.val),
        "mkl_sparse_create_csr");

    try {
      detail::CheckMklSparseStatus(mkl_sparse_set_mv_hint(handle_, SPARSE_OPERATION_NON_TRANSPOSE,
                                                          descriptor_, expected_calls),
                                   "mkl_sparse_set_mv_hint");
      detail::CheckMklSparseStatus(mkl_sparse_optimize(handle_), "mkl_sparse_optimize");
    } catch (...) {
      mkl_sparse_destroy(handle_);
      handle_ = nullptr;
      throw;
    }
  }

  MklCsrSpmv(const MklCsrSpmv&) = delete;
  MklCsrSpmv& operator=(const MklCsrSpmv&) = delete;

  MklCsrSpmv(MklCsrSpmv&& rhs) noexcept : handle_(rhs.handle_) { rhs.handle_ = nullptr; }

  MklCsrSpmv& operator=(MklCsrSpmv&& rhs) noexcept
  {
    if (this != &rhs) {
      if (handle_ != nullptr) {
        mkl_sparse_destroy(handle_);
      }
      handle_ = rhs.handle_;
      rhs.handle_ = nullptr;
    }
    return *this;
  }

  ~MklCsrSpmv()
  {
    if (handle_ != nullptr) {
      mkl_sparse_destroy(handle_);
    }
  }

  void Multiply(const NT* x, NT* y) const
  {
    detail::CheckMklSparseStatus(detail::MklSparseMv<NT>(handle_, descriptor_, x, y),
                                 "mkl_sparse_mv");
  }

 private:
  sparse_matrix_t handle_ = nullptr;
  matrix_descr descriptor_{SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT};
};

}  // namespace spcraft

#endif  // SPCRAFT_USE_MKL
