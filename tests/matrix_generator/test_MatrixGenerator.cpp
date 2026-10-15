#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>

#include "SpCraft.h"

namespace spcraft::tests
{
namespace
{

template <class Matrix>
bool IsValidUndirectedGraph(const Matrix& graph)
{
  if (graph.m != graph.n || graph.row_ptr[0] != 0 || graph.row_ptr[graph.m] != graph.nnz) {
    return false;
  }

  for (int row = 0; row < graph.m; ++row) {
    if (!std::is_sorted(graph.col_id + graph.row_ptr[row], graph.col_id + graph.row_ptr[row + 1])) {
      return false;
    }
    for (int position = graph.row_ptr[row]; position < graph.row_ptr[row + 1]; ++position) {
      const int column = graph.col_id[position];
      if (column == row || graph.val[position] != 1.0) {
        return false;
      }
      if (!std::binary_search(graph.col_id + graph.row_ptr[column], graph.col_id + graph.row_ptr[column + 1],
                              row)) {
        return false;
      }
    }
  }
  return true;
}

TEST(MatrixGenerator, ReproducibleUndirectedRmat)
{
  auto first = spcraft::GenRMAT<double>(6, 8, 42);
  auto second = spcraft::GenRMAT<double>(6, 8, 42);

  EXPECT_EQ(first.m, 64);
  EXPECT_EQ(first.n, 64);
  EXPECT_GT(first.nnz, 0);
  EXPECT_TRUE(IsValidUndirectedGraph(first));
  ASSERT_EQ(first.nnz, second.nnz);
  const auto first_rows_end = first.row_ptr + first.m + 1;
  const auto first_columns_end = first.col_id + first.nnz;
  EXPECT_TRUE(std::equal(first.row_ptr, first_rows_end, second.row_ptr));
  EXPECT_TRUE(std::equal(first.col_id, first_columns_end, second.col_id));
}
}  // namespace
}  // namespace spcraft::tests
