#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Generate an undirected Erdős–Rényi graph in CSR format.
 *
 * Uses the Batagelj–Brandes edge-skipping algorithm to sample G(n, p)
 * without examining all O(n²) possible edges.
 */
template <class NT, class IT, class OT = IT>
[[nodiscard]] CsrMatrix<IT, NT, OT> GenERGraph(IT vertices, double expected_degree,
                                               std::uint64_t seed)
{
  if (vertices < 2 || expected_degree <= 0.0 || expected_degree >= vertices - 1.0) {
    throw std::invalid_argument("expected degree must be in (0, vertices - 1)");
  }

  const double probability = expected_degree / static_cast<double>(vertices - 1);
  const double log_one_minus_p = std::log1p(-probability);
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  std::vector<std::pair<IT, IT>> edges;
  edges.reserve(static_cast<std::size_t>(vertices * expected_degree * 0.525));

  OT source = 1;
  OT destination = -1;
  while (source < vertices) {
    const double random_value = uniform(generator);
    destination +=
        1 + static_cast<OT>(std::floor(std::log1p(-random_value) / log_one_minus_p));
    while (destination >= source && source < vertices) {
      destination -= source;
      ++source;
    }
    if (source < vertices) {
      edges.emplace_back(static_cast<IT>(source), static_cast<IT>(destination));
    }
  }

  std::vector<OT> degrees(static_cast<std::size_t>(vertices), 0);
  for (const auto& [row, column] : edges) {
    ++degrees[row];
    ++degrees[column];
  }

  CsrMatrix<IT, NT, OT> graph;
  const OT nonzeros = static_cast<OT>(edges.size()) * 2;
  graph.Allocate(nonzeros, vertices, vertices);
  graph.row_ptr[0] = 0;
  for (IT row = 0; row < vertices; ++row) {
    graph.row_ptr[row + 1] = graph.row_ptr[row] + degrees[row];
  }

  std::vector<OT> next(graph.row_ptr, graph.row_ptr + vertices);
  for (const auto& [row, column] : edges) {
    const OT forward = next[row]++;
    const OT reverse = next[column]++;
    graph.col_id[forward] = column;
    graph.col_id[reverse] = row;
    graph.val[forward] = static_cast<NT>(1.0);
    graph.val[reverse] = static_cast<NT>(1.0);
  }
  return graph;
}

}  // namespace spcraft
