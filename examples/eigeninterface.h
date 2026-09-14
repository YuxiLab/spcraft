#pragma once
#include "SpCraft.h"
#include "Eigen/Sparse"
#include "core/CsrMatrix.h"
namespace spcraft
{
// Mutable view: allows writing directly into vec.val
template <class IT, class NT>
Eigen::Map<Eigen::Matrix<NT, Eigen::Dynamic, 1>> ToEigen(DenseVector<IT, NT>& vec)
{
  return Eigen::Map<Eigen::Matrix<NT, Eigen::Dynamic, 1>>(vec.val,
                                                          static_cast<Eigen::Index>(vec.n));
}

// Read-only view: enforces const-correctness
template <class IT, class NT>
Eigen::Map<const Eigen::Matrix<NT, Eigen::Dynamic, 1>> ToEigen(const DenseVector<IT, NT>& vec)
{
  return Eigen::Map<const Eigen::Matrix<NT, Eigen::Dynamic, 1>>(vec.val,
                                                                static_cast<Eigen::Index>(vec.n));
}

// Mutable CSR view: allows modifying existing non-zero values in-place
template <class IT, class NT, class OT = IT>
Eigen::Map<Eigen::SparseMatrix<NT, Eigen::RowMajor, IT>> ToEigen(CsrMatrix<IT, NT, OT>& csr)
{
  return Eigen::Map<Eigen::SparseMatrix<NT, Eigen::RowMajor, IT>>(
      static_cast<Eigen::Index>(csr.m), static_cast<Eigen::Index>(csr.n),
      static_cast<Eigen::Index>(csr.nnz),
      csr.row_ptr,  // Outer starts: size nrows + 1
      csr.col_id,   // Inner indices: size nnz
      csr.val,      // Values: size nnz
      nullptr       // Inner non-zeros array: nullptr means fully compressed
  );
}

// Read-only CSR view: enforces const-correctness
template <class IT, class NT, class OT = IT>
Eigen::Map<const Eigen::SparseMatrix<NT, Eigen::RowMajor, IT>> ToEigen(
    const CsrMatrix<IT, NT, OT>& csr)
{
  return Eigen::Map<const Eigen::SparseMatrix<NT, Eigen::RowMajor, IT>>(
      static_cast<Eigen::Index>(csr.m), static_cast<Eigen::Index>(csr.n),
      static_cast<Eigen::Index>(csr.nnz), csr.row_ptr, csr.col_id, csr.val, nullptr);
}

}  // namespace spcraft