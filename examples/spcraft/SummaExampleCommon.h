#pragma once

#include "SpCraft.h"
#include "DistributedExampleCommon.h"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace spcraft;

//! Select owned entries, then convert only local storage.
using ExampleGrid = std::shared_ptr<const CommGrid2D>;

template <class IT, class NT, class OT, class Predicate>
DISTSPCSC PartitionSummaInput(const SPCOO& source, ExampleGrid grid, Predicate owns)
{
  DistSpCOO<IT, NT, OT> selected(grid, source.m, source.n);
  OT count = 0;
  for (OT p = 0; p < source.nnz; ++p) {
    const TupleEntry<IT, NT>& entry = source.entries[p];
    if (owns(entry.row, entry.col)) ++count;
  }
  selected.Allocate(count);
  OT position = 0;
  for (OT p = 0; p < source.nnz; ++p) {
    const TupleEntry<IT, NT>& entry = source.entries[p];
    if (!owns(entry.row, entry.col)) continue;
    selected.local.entries[position++] = {selected.LocalRow(entry.row), selected.LocalCol(entry.col),
                                          entry.val};
  }
  CscMatrix<IT, NT, OT> local = selected.local.ToCsc();
  DistSpCSC<IT, NT, OT> result(grid, std::move(local), source.m, source.n);
  return result;
}

//! Compare a packed local result with the corresponding global Eigen entries.
template <class IT, class NT, class OT>
bool CheckSummaReference(const DistSpCOO<IT, NT, OT>& result, const ExampleSparseMatrix<IT, NT>& expected)
{
  const ExampleSparseMatrix<IT, NT> actual = ToEigenReference(result.local);
  std::vector<Eigen::Triplet<NT, IT>> entries;
  for (IT col = 0; col < expected.outerSize(); ++col) {
    for (typename ExampleSparseMatrix<IT, NT>::InnerIterator entry(expected, col); entry; ++entry) {
      const IT row = static_cast<IT>(entry.row());
      if (result.Owns(row, col)) {
        entries.emplace_back(result.LocalRow(row), result.LocalCol(col), entry.value());
      }
    }
  }
  ExampleSparseMatrix<IT, NT> local_reference(result.local.m, result.local.n);
  local_reference.setFromTriplets(entries.begin(), entries.end());
  const ExampleSparseMatrix<IT, NT> error = actual - local_reference;
  const NT tolerance = NT{100} * std::numeric_limits<NT>::epsilon();
  return error.norm() <= tolerance * std::max(NT{1}, local_reference.norm());
}
