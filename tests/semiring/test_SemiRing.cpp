#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <iostream>

#include "SpCraft.h"

namespace spcraft::tests
{
namespace
{

static_assert(spcraft::PlusTimesRing<int>::kAdditiveIdentity == 0);
static_assert(spcraft::PlusTimesRing<int>::kMultiplicativeIdentity == 1);
static_assert(spcraft::PlusTimesRing<int>::Add(2, 3) == 5);
static_assert(spcraft::PlusTimesRing<int>::Multiply(2, 3) == 6);

using OrAndRing = spcraft::PlusTimesRing<bool>;

static_assert(!OrAndRing::kAdditiveIdentity);
static_assert(OrAndRing::kMultiplicativeIdentity);
static_assert(OrAndRing::Add(false, true));
static_assert(!OrAndRing::Multiply(true, false));
static_assert(OrAndRing::Multiply(true, true));

TEST(Semiring, BfsFrontierExpansion)
{
  // A(destination, source): vertex 0 has outgoing edges to vertices 1 and 2.
  spcraft::CsrMatrix<int, bool> adjacency;
  adjacency.Allocate(2, 3, 3);
  constexpr std::array<int, 4> row_offsets{0, 0, 1, 2};
  constexpr std::array<int, 2> column_indices{0, 0};
  std::copy(row_offsets.begin(), row_offsets.end(), adjacency.row_ptr);
  std::copy(column_indices.begin(), column_indices.end(), adjacency.col_id);
  std::fill(adjacency.val, adjacency.val + adjacency.nnz, true);

  spcraft::DenseVector<int, bool> frontier(3);
  frontier.val[0] = true;
  spcraft::DenseVector<int, bool> next(3);
  constexpr std::array<bool, 3> expected{false, true, true};

  spcraft::OmpSpMV<OrAndRing>(adjacency, frontier, next);

  for (std::size_t row = 0; row < expected.size(); ++row) {
    EXPECT_EQ(next.val[row], expected[row]) << "vertex " << row;
  }
}
}  // namespace
}  // namespace spcraft::tests
