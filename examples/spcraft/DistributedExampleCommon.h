#pragma once

#include "SpCraft.h"

#include <vector>

#include <Eigen/SparseCore>

using namespace spcraft;

template <class IT, class NT>
using ExampleSparseMatrix = Eigen::SparseMatrix<NT, Eigen::ColMajor, IT>;

template <class IT, class NT, class OT>
ExampleSparseMatrix<IT, NT> ToEigenReference(const CooMatrix<IT, NT, OT>& source)
{
  std::vector<Eigen::Triplet<NT, IT>> entries;
  entries.reserve(source.nnz);
  for (OT position = 0; position < source.nnz; ++position) {
    const TupleEntry<IT, NT>& entry = source.entries[position];
    entries.emplace_back(entry.row, entry.col, entry.val);
  }
  ExampleSparseMatrix<IT, NT> result(source.m, source.n);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

template <class IT, class NT, class OT>
ExampleSparseMatrix<IT, NT> ToEigenReference(const CsrMatrix<IT, NT, OT>& source)
{
  std::vector<Eigen::Triplet<NT, IT>> entries;
  entries.reserve(source.nnz);
  for (IT row = 0; row < source.m; ++row) {
    for (OT position = source.row_ptr[row]; position < source.row_ptr[row + 1]; ++position) {
      entries.emplace_back(row, source.col_id[position], source.val[position]);
    }
  }
  ExampleSparseMatrix<IT, NT> result(source.m, source.n);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

template <class IT, class NT, class OT>
ExampleSparseMatrix<IT, NT> ToEigenReference(const CscMatrix<IT, NT, OT>& source)
{
  std::vector<Eigen::Triplet<NT, IT>> entries;
  entries.reserve(source.nnz);
  for (IT col = 0; col < source.n; ++col) {
    for (OT position = source.col_ptr[col]; position < source.col_ptr[col + 1]; ++position) {
      entries.emplace_back(source.row_id[position], col, source.val[position]);
    }
  }
  ExampleSparseMatrix<IT, NT> result(source.m, source.n);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

template <class IT, class NT, class OT>
ExampleSparseMatrix<IT, NT> ToEigenReference(const DcscMatrix<IT, NT, OT>& source)
{
  std::vector<Eigen::Triplet<NT, IT>> entries;
  entries.reserve(source.nnz);
  for (IT stored = 0; stored < source.nzc; ++stored) {
    const IT col = source.col_id[stored];
    for (OT position = source.col_ptr[stored]; position < source.col_ptr[stored + 1]; ++position) {
      entries.emplace_back(source.row_id[position], col, source.val[position]);
    }
  }
  ExampleSparseMatrix<IT, NT> result(source.m, source.n);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

template <class IT, class NT, class OT>
bool CheckDistributedReference(const DistSpCOO<IT, NT, OT>& matrix, const ExampleSparseMatrix<IT, NT>& global)
{
  const ExampleSparseMatrix<IT, NT> actual = ToEigenReference(matrix.local);
  int64_t expected_nnz = 0;
  for (IT col = 0; col < global.outerSize(); ++col) {
    for (typename ExampleSparseMatrix<IT, NT>::InnerIterator entry(global, col); entry; ++entry) {
      const IT row = static_cast<IT>(entry.row());
      const IT column = static_cast<IT>(entry.col());
      if (!matrix.Owns(row, column)) continue;
      ++expected_nnz;
      if (actual.coeff(matrix.LocalRow(row), matrix.LocalCol(column)) != entry.value()) return false;
    }
  }
  return actual.nonZeros() == expected_nnz;
}
