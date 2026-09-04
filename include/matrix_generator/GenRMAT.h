#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/CsrMatrix.h"

namespace spcraft
{

/**
 * @brief Generate an undirected R-MAT graph in CSR format.
 *
 * Samples edge_factor * 2^scale edge tuples using recursive quadrant
 * selection. The defaults are the Graph500 initiator weights. Self-loops and
 * duplicate edges are removed before the graph is symmetrized.
 */
template <class NT, class IT, class OT = IT>
[[nodiscard]] CsrMatrix<IT, NT, OT> GenRMAT(IT scale, std::size_t edge_factor,
                                            std::uint64_t seed, double a = 0.57,
                                            double b = 0.19, double c = 0.19, double d = 0.05)
{
  static_assert(std::is_integral_v<IT>, "R-MAT index type must be integral");
  static_assert(std::is_integral_v<OT>, "R-MAT offset type must be integral");

  if constexpr (std::is_signed_v<IT>) {
    if (scale <= 0) {
      throw std::invalid_argument("R-MAT scale must be positive");
    }
  } else if (scale == 0) {
    throw std::invalid_argument("R-MAT scale must be positive");
  }

  using UnsignedIndex = std::make_unsigned_t<IT>;
  const auto scale_value = static_cast<std::uintmax_t>(scale);
  if (scale_value >= std::numeric_limits<IT>::digits) {
    throw std::invalid_argument("R-MAT scale does not fit the index type");
  }
  if (edge_factor == 0) {
    throw std::invalid_argument("R-MAT edge factor must be positive");
  }

  const double total_weight = a + b + c + d;
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d) ||
      a < 0.0 || b < 0.0 || c < 0.0 || d < 0.0 || total_weight <= 0.0) {
    throw std::invalid_argument("R-MAT initiator weights must be finite and nonnegative");
  }

  const auto vertices_unsigned = static_cast<UnsignedIndex>(UnsignedIndex{1} << scale_value);
  if constexpr (sizeof(UnsignedIndex) > sizeof(std::size_t)) {
    if (vertices_unsigned > std::numeric_limits<std::size_t>::max()) {
      throw std::length_error("R-MAT vertex count exceeds addressable memory");
    }
  }
  const auto vertex_count = static_cast<std::size_t>(vertices_unsigned);
  if (vertex_count > std::numeric_limits<std::size_t>::max() / edge_factor) {
    throw std::length_error("R-MAT edge count exceeds addressable memory");
  }
  const auto sample_count = vertex_count * edge_factor;

  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> uniform(0.0, total_weight);
  std::vector<std::pair<IT, IT>> edges;
  edges.reserve(sample_count);

  const double ab = a + b;
  const double abc = ab + c;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    UnsignedIndex source = 0;
    UnsignedIndex destination = 0;

    for (std::uintmax_t level = 0; level < scale_value; ++level) {
      const auto bit =
          static_cast<UnsignedIndex>(UnsignedIndex{1} << (scale_value - level - 1));
      const double quadrant = uniform(generator);
      if (quadrant >= ab) {
        source |= bit;
      }
      if ((quadrant >= a && quadrant < ab) || quadrant >= abc) {
        destination |= bit;
      }
    }

    if (source != destination) {
      const IT first = static_cast<IT>(std::min(source, destination));
      const IT second = static_cast<IT>(std::max(source, destination));
      edges.emplace_back(first, second);
    }
  }

  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  if (edges.empty()) {
    throw std::runtime_error("R-MAT sampling produced no non-self edges");
  }

  const auto max_offset = static_cast<std::uintmax_t>(std::numeric_limits<OT>::max());
  if (edges.size() > max_offset / 2) {
    throw std::length_error("R-MAT nonzero count does not fit the offset type");
  }

  const IT vertices = static_cast<IT>(vertices_unsigned);
  std::vector<OT> degrees(vertex_count, 0);
  for (const auto& [source, destination] : edges) {
    ++degrees[static_cast<std::size_t>(source)];
    ++degrees[static_cast<std::size_t>(destination)];
  }

  CsrMatrix<IT, NT, OT> graph;
  const OT nonzeros = static_cast<OT>(edges.size() * 2);
  graph.Allocate(nonzeros, vertices, vertices);
  graph.row_ptr[0] = 0;
  for (IT row = 0; row < vertices; ++row) {
    graph.row_ptr[row + 1] = graph.row_ptr[row] + degrees[static_cast<std::size_t>(row)];
  }

  std::vector<OT> next(graph.row_ptr, graph.row_ptr + vertices);
  for (const auto& [source, destination] : edges) {
    const OT forward = next[static_cast<std::size_t>(source)]++;
    const OT reverse = next[static_cast<std::size_t>(destination)]++;
    graph.col_id[forward] = destination;
    graph.col_id[reverse] = source;
    graph.val[forward] = static_cast<NT>(1);
    graph.val[reverse] = static_cast<NT>(1);
  }

  for (IT row = 0; row < vertices; ++row) {
    std::sort(graph.col_id + graph.row_ptr[row], graph.col_id + graph.row_ptr[row + 1]);
  }
  return graph;
}

}  // namespace spcraft
