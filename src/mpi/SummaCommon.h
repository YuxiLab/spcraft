#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "kernel/mtSpGEMM.h"
#include "mpi/2dspgemm.h"
#include "mpi/mpiutil.h"

namespace spcraft::detail
{

template <class T>
void BroadcastArray(T* data, int64_t count, int root, MPI_Comm comm)
{
  const int entries = count;
  MPI_Bcast(data, entries, MPIType<T>(), root, comm);
}

//! Traverse logical columns without allocating dense headers for DCSC inputs.
template <class IT, class NT, class OT, class Function>
void VisitColumns(const CscMatrix<IT, NT, OT>& matrix, Function visit)
{
  for (IT column = 0; column < matrix.n; ++column) {
    visit(column, matrix.col_ptr[column], matrix.col_ptr[column + 1]);
  }
}

template <class IT, class NT, class OT, class Function>
void VisitColumns(const DcscMatrix<IT, NT, OT>& matrix, Function visit)
{
  for (IT column = 0; column < matrix.nzc; ++column) {
    visit(matrix.col_id[column], matrix.col_ptr[column], matrix.col_ptr[column + 1]);
  }
}

//! The A panel keeps packed output rows and rebases one inner block's columns.
template <class IT, class NT, class OT, class Matrix>
DcscMatrix<IT, NT, OT> ColumnPanel(const Matrix& source, IT first, IT width)
{
  OT count = 0;
  IT columns = 0;
  VisitColumns(source, [&](IT column, OT begin, OT end) {
    if (column >= first && column - first < width && begin != end) {
      count += end - begin;
      ++columns;
    }
  });
  DcscMatrix<IT, NT, OT> panel;
  panel.Allocate(count, source.m, width, columns);
  OT position = 0;
  IT stored = 0;
  VisitColumns(source, [&](IT column, OT begin, OT end) {
    if (column < first || column - first >= width || begin == end) return;
    panel.col_id[stored] = column - first;
    panel.col_ptr[stored++] = position;
    for (OT p = begin; p < end; ++p) {
      panel.row_id[position] = source.row_id[p];
      panel.val[position++] = source.val[p];
    }
  });
  panel.col_ptr[stored] = position;
  return panel;
}

//! The B panel rebases inner rows while preserving packed output columns.
template <class IT, class NT, class OT, class Matrix>
DcscMatrix<IT, NT, OT> RowPanel(const Matrix& source, IT first, IT width)
{
  OT count = 0;
  IT columns = 0;
  VisitColumns(source, [&](IT column, OT begin, OT end) {
    bool occupied = false;
    for (OT p = begin; p < end; ++p) {
      const IT row = source.row_id[p];
      if (row >= first && row - first < width) {
        ++count;
        occupied = true;
      }
    }
    if (occupied) ++columns;
  });
  DcscMatrix<IT, NT, OT> panel;
  panel.Allocate(count, width, source.n, columns);
  OT position = 0;
  IT stored = 0;
  VisitColumns(source, [&](IT column, OT begin, OT end) {
    const OT start = position;
    for (OT p = begin; p < end; ++p) {
      const IT row = source.row_id[p];
      if (row >= first && row - first < width) {
        panel.row_id[position] = row - first;
        panel.val[position++] = source.val[p];
      }
    }
    if (position != start) {
      panel.col_id[stored] = column;
      panel.col_ptr[stored++] = start;
    }
  });
  panel.col_ptr[stored] = position;
  return panel;
}

template <class IT, class NT, class OT>
void BroadcastPanel(DcscMatrix<IT, NT, OT>& panel, int root, MPI_Comm comm, IT m, IT n)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  OT count = panel.nnz;
  IT columns = panel.nzc;
  MPI_Bcast(&count, 1, MPIType<OT>(), root, comm);
  MPI_Bcast(&columns, 1, MPIType<IT>(), root, comm);
  if (rank != root) panel.Allocate(count, m, n, columns);
  BroadcastArray(panel.col_ptr, int64_t{1} + columns, root, comm);
  BroadcastArray(panel.col_id, columns, root, comm);
  BroadcastArray(panel.row_id, count, root, comm);
  BroadcastArray(panel.val, count, root, comm);
}

//! Merge sorted, unique COO entries; retain structural zeros after cancellation.
template <class SemiRing, class IT, class NT, class OT>
CooMatrix<IT, NT, OT> AddProducts(const CooMatrix<IT, NT, OT>& a, const CooMatrix<IT, NT, OT>& b)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>);
  const auto less = [](const TupleEntry<IT, NT>& x, const TupleEntry<IT, NT>& y) {
    return x.col < y.col || (x.col == y.col && x.row < y.row);
  };

  // Count the union before narrowing: intermediate products can overlap heavily.
  OT i = 0;
  OT j = 0;
  int64_t count = 0;
  while (i < a.nnz || j < b.nnz) {
    if (j == b.nnz || (i < a.nnz && less(a.entries[i], b.entries[j])))
      ++i;
    else if (i == a.nnz || less(b.entries[j], a.entries[i]))
      ++j;
    else {
      ++i;
      ++j;
    }
    ++count;
  }
  if (count > std::numeric_limits<OT>::max()) {
    throw std::overflow_error("SUMMA output nnz exceeds offset type limit");
  }
  CooMatrix<IT, NT, OT> result;
  result.Allocate(count, a.m, a.n);
  i = 0;
  j = 0;
  OT position = 0;
  while (i < a.nnz || j < b.nnz) {
    if (j == b.nnz || (i < a.nnz && less(a.entries[i], b.entries[j]))) {
      result.entries[position++] = a.entries[i++];
    } else if (i == a.nnz || less(b.entries[j], a.entries[i])) {
      result.entries[position++] = b.entries[j++];
    } else {
      TupleEntry<IT, NT> entry = a.entries[i++];
      entry.val = SemiRing::Add(entry.val, b.entries[j++].val);
      result.entries[position++] = entry;
    }
  }
  return result;
}

template <class Matrix>
void CheckSummaInputs(const Matrix& A, const Matrix& B)
{
  const bool same_grid = A.commgrid && B.commgrid && A.commgrid == B.commgrid;
  if (!same_grid || A.commgrid->World() == MPI_COMM_NULL) {
    throw std::invalid_argument("SUMMA inputs must share the same live communicator grid");
  }
  if (A.commgrid->Rows() != A.commgrid->Cols()) {
    throw std::invalid_argument("SUMMA currently requires a square process grid");
  }
  if (A.n != B.m) {
    throw std::invalid_argument("SUMMA inputs need compatible inner dimensions");
  }
  const bool a_matches = A.local.m == A.LocalRows() && A.local.n == A.LocalCols();
  const bool b_matches = B.local.m == B.LocalRows() && B.local.n == B.LocalCols();
  if (!a_matches || !b_matches) {
    throw std::invalid_argument("SUMMA local dimensions do not match the process-grid partition");
  }
}

//! Every layer owns blocks layer, layer+layers, ...; 2D uses layer=0, layers=1.
template <class SemiRing, class IT, class NT, class OT, class Matrix>
DISTSPCOO RunSummaStages(const Matrix& A, const Matrix& B, int layer, int layers)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>);
  CheckSummaInputs(A, B);
  DistSpCOO<IT, NT, OT> result(A.commgrid, A.m, B.n);
  const CommGrid2D& grid = *A.commgrid;
  const int64_t inner = A.n;
  if (inner == 0) return result;
  const int64_t block = inner / grid.Cols() + (inner % grid.Cols() != 0);
  const int64_t blocks = inner / block + (inner % block != 0);

  // Blocking broadcasts finish before local multiplication or the next panel.
  for (int64_t k = layer; k < blocks; k += layers) {
    // @profiler-iter k start
    // @profiler-region stage start
    const IT width = std::min(block, inner - k * block);
    const int root = k;
    const IT first = 0;
    DcscMatrix<IT, NT, OT> a_panel;
    DcscMatrix<IT, NT, OT> b_panel;
    // @profiler-region panel_preparation start
    if (grid.Col() == root) a_panel = ColumnPanel<IT, NT, OT>(A.local, first, width);
    if (grid.Row() == root) b_panel = RowPanel<IT, NT, OT>(B.local, first, width);
    // @profiler-region panel_preparation end
    // @profiler-region bcast_A start
    BroadcastPanel(a_panel, root, grid.RowWorld(), A.local.m, width);
    // @profiler-region bcast_A end
    // @profiler-region bcast_B start
    BroadcastPanel(b_panel, root, grid.ColWorld(), width, B.local.n);
    // @profiler-region bcast_B end
    // @profiler-region local_spgemm start
    CooMatrix<IT, NT, OT> partial = OmpHashSpGEMM<SemiRing>(a_panel, b_panel);
    // @profiler-region local_spgemm end
    // @profiler-region merge start
    result.local = AddProducts<SemiRing>(result.local, partial);
    // @profiler-region merge end
    // @profiler-region stage end
    // @profiler-iter k end
  }
  return result;
}

}  // namespace spcraft::detail
