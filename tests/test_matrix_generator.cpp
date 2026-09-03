#include <algorithm>
#include <iostream>

#include "MatrixGenerator.h"

namespace
{

template <class Matrix>
bool IsValidUndirectedGraph(const Matrix& graph)
{
  if (graph.m != graph.n || graph.row_ptr[0] != 0 || graph.row_ptr[graph.m] != graph.nnz) {
    return false;
  }

  for (int row = 0; row < graph.m; ++row) {
    if (!std::is_sorted(graph.col_id + graph.row_ptr[row],
                        graph.col_id + graph.row_ptr[row + 1])) {
      return false;
    }
    for (int position = graph.row_ptr[row]; position < graph.row_ptr[row + 1]; ++position) {
      const int column = graph.col_id[position];
      if (column == row || graph.val[position] != 1.0) {
        return false;
      }
      if (!std::binary_search(graph.col_id + graph.row_ptr[column],
                              graph.col_id + graph.row_ptr[column + 1], row)) {
        return false;
      }
    }
  }
  return true;
}

bool TestRMAT()
{
  auto first = spcraft::GenRMAT<double>(6, 8, 42);
  auto second = spcraft::GenRMAT<double>(6, 8, 42);

  if (first.m != 64 || first.n != 64 || first.nnz == 0 || !IsValidUndirectedGraph(first)) {
    std::cerr << "R-MAT generated an invalid CSR graph\n";
    return false;
  }
  if (first.nnz != second.nnz ||
      !std::equal(first.row_ptr, first.row_ptr + first.m + 1, second.row_ptr) ||
      !std::equal(first.col_id, first.col_id + first.nnz, second.col_id)) {
    std::cerr << "R-MAT is not reproducible for a fixed seed\n";
    return false;
  }
  return true;
}

}  // namespace

int main()
{
  return TestRMAT() ? 0 : 1;
}
