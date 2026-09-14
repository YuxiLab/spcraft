#pragma once

#include "utils/omp/omp_wrapper.h"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "core/CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Iteration controls shared by every PageRank backend.
 */
template <class NT>
struct PageRankOptions {
  NT damping = NT{0.85};     //!< Probability of following an edge; must be in [0, 1).
  NT tolerance = NT{1e-6};   //!< Convergence threshold on the L1 change per iteration.
  int max_iterations = 100;  //!< Upper bound on power iterations.
};

/**
 * @brief What a PageRank run did.
 */
template <class NT>
struct PageRankResult {
  int iterations = 0;      //!< Power iterations actually performed.
  NT residual = NT{0};     //!< L1 change over the final iteration.
  bool converged = false;  //!< Whether residual fell to or below the tolerance.
};

/**
 * @brief Build the transposed, out-edge-normalized operator PageRank iterates on.
 *
 * Takes the adjacency matrix in CSR, where row i holds the out-edges of vertex i,
 * and returns P^T in CSR: row j holds one entry per in-edge (i, j) carrying
 * A(i, j) divided by the sum of row i. For a 0/1 adjacency that value is
 * 1 / outdeg(i), which is the textbook random-surfer transition probability;
 * a weighted graph gets the weighted equivalent.
 *
 * Transposing is what lets the iteration run as a plain SpMV: the rank update
 * needs the sum over in-neighbours, and CSR only walks out-neighbours cheaply.
 * The result is a row-stochastic-by-columns matrix, so it is built once and
 * reused across every iteration and every backend.
 *
 * Dangling vertices -- rows summing to zero, including empty rows -- contribute
 * no entries at all. Their rank is redistributed during the iteration instead,
 * which is why the operator alone does not determine the answer.
 *
 * @param adjacency Square CSR adjacency matrix with non-negative values.
 * @return P^T in CSR, with at most adjacency.nnz entries.
 */
template <class IT, class NT, class OT>
[[nodiscard]] CsrMatrix<IT, NT, OT> pagerank_operator(const CsrMatrix<IT, NT, OT>& adjacency)
{
  static_assert(std::is_floating_point_v<NT>, "PageRank requires a floating-point value type");
  static_assert(std::is_integral_v<IT>, "PageRank index type must be integral");
  static_assert(std::is_integral_v<OT>, "PageRank offset type must be integral");

  if (adjacency.m != adjacency.n) {
    throw std::invalid_argument("PageRank requires a square adjacency matrix");
  }
  if constexpr (std::is_signed_v<IT>) {
    if (adjacency.m < 0) {
      throw std::invalid_argument("PageRank vertex count must be non-negative");
    }
  }

  const IT n = adjacency.m;
  CsrMatrix<IT, NT, OT> result;
  if (n == IT{0}) {
    result.Allocate(OT{0}, n, n);
    return result;
  }

  // Row sums double as the out-degree of each vertex; a zero sum marks a
  // dangling vertex whose edges are dropped from the operator.
  std::vector<NT> row_sum(static_cast<std::size_t>(n), NT{0});
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(adjacency, row_sum, n))
  for (IT row = 0; row < n; ++row) {
    NT sum = NT{0};
    for (OT position = adjacency.row_ptr[row]; position < adjacency.row_ptr[row + 1];
         ++position) {
      sum += adjacency.val[position];
    }
    row_sum[static_cast<std::size_t>(row)] = sum;
  }

  // Counting pass over the transpose: how many in-edges does each vertex keep?
  std::vector<OT> counts(static_cast<std::size_t>(n) + 1, OT{0});
  OT kept = OT{0};
  for (IT row = 0; row < n; ++row) {
    if (row_sum[static_cast<std::size_t>(row)] <= NT{0}) continue;
    for (OT position = adjacency.row_ptr[row]; position < adjacency.row_ptr[row + 1];
         ++position) {
      const IT column = adjacency.col_id[position];
      if (column < IT{0} || column >= n) {
        throw std::invalid_argument("PageRank adjacency column index is out of range");
      }
      ++counts[static_cast<std::size_t>(column) + 1];
      ++kept;
    }
  }

  result.Allocate(kept, n, n);
  for (IT row = 0; row < n; ++row) {
    counts[static_cast<std::size_t>(row) + 1] += counts[static_cast<std::size_t>(row)];
    result.row_ptr[row] = counts[static_cast<std::size_t>(row)];
  }
  result.row_ptr[n] = counts[static_cast<std::size_t>(n)];

  // Scatter pass: place each surviving edge into its column's row of P^T.
  std::vector<OT> offsets(counts.begin(), counts.begin() + static_cast<std::size_t>(n));
  for (IT row = 0; row < n; ++row) {
    const NT sum = row_sum[static_cast<std::size_t>(row)];
    if (sum <= NT{0}) continue;
    for (OT position = adjacency.row_ptr[row]; position < adjacency.row_ptr[row + 1];
         ++position) {
      const IT column = adjacency.col_id[position];
      const OT destination = offsets[static_cast<std::size_t>(column)]++;
      result.col_id[destination] = row;
      result.val[destination] = adjacency.val[position] / sum;
    }
  }

  return result;
}

namespace detail
{

//! Reject option sets no backend can act on. Shared so both backends agree.
template <class NT>
[[nodiscard]] constexpr bool PageRankOptionsAreValid(const PageRankOptions<NT>& options)
{
  return options.damping >= NT{0} && options.damping < NT{1} && options.tolerance >= NT{0} &&
         options.max_iterations >= 0;
}

}  // namespace detail

}  // namespace spcraft
