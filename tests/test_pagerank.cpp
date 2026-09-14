#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "SpCraft.h"

namespace
{

using Index = std::int32_t;
using Value = double;
using Offset = std::int64_t;
using Matrix = spcraft::CsrMatrix<Index, Value, Offset>;
using Vector = spcraft::DenseVector<Index, Value>;

//! Build a CSR adjacency matrix from an edge list, values all 1.
Matrix FromEdges(Index vertices, const std::vector<std::pair<Index, Index>>& edges)
{
  std::vector<Offset> row_ptr(static_cast<std::size_t>(vertices) + 1, 0);
  for (const auto& [from, to] : edges) {
    (void)to;
    ++row_ptr[static_cast<std::size_t>(from) + 1];
  }
  for (Index vertex = 0; vertex < vertices; ++vertex) {
    row_ptr[static_cast<std::size_t>(vertex) + 1] += row_ptr[static_cast<std::size_t>(vertex)];
  }

  Matrix matrix;
  matrix.Allocate(static_cast<Offset>(edges.size()), vertices, vertices);
  std::copy(row_ptr.begin(), row_ptr.end(), matrix.row_ptr);

  std::vector<Offset> cursor(row_ptr.begin(), row_ptr.end() - 1);
  for (const auto& [from, to] : edges) {
    const Offset destination = cursor[static_cast<std::size_t>(from)]++;
    matrix.col_id[destination] = to;
    matrix.val[destination] = 1.0;
  }
  return matrix;
}

bool Close(Value left, Value right, Value tolerance = 1e-9)
{
  return std::abs(left - right) <= tolerance;
}

//! Reference power iteration written straight from the definition, no SpMV.
std::vector<Value> ReferencePageRank(Index vertices,
                                     const std::vector<std::pair<Index, Index>>& edges,
                                     Value damping, int iterations)
{
  std::vector<Value> out_degree(static_cast<std::size_t>(vertices), 0.0);
  for (const auto& [from, to] : edges) {
    (void)to;
    out_degree[static_cast<std::size_t>(from)] += 1.0;
  }

  const Value uniform = 1.0 / static_cast<Value>(vertices);
  std::vector<Value> rank(static_cast<std::size_t>(vertices), uniform);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::vector<Value> next(static_cast<std::size_t>(vertices), 0.0);
    Value dangling = 0.0;
    for (Index vertex = 0; vertex < vertices; ++vertex) {
      if (out_degree[static_cast<std::size_t>(vertex)] == 0.0) {
        dangling += rank[static_cast<std::size_t>(vertex)];
      }
    }
    for (const auto& [from, to] : edges) {
      next[static_cast<std::size_t>(to)] +=
          rank[static_cast<std::size_t>(from)] / out_degree[static_cast<std::size_t>(from)];
    }
    for (Index vertex = 0; vertex < vertices; ++vertex) {
      next[static_cast<std::size_t>(vertex)] =
          (1.0 - damping) * uniform +
          damping * (next[static_cast<std::size_t>(vertex)] + dangling * uniform);
    }
    rank = next;
  }
  return rank;
}

}  // namespace

int main()
{
  int failures = 0;

  // ---------------------------------------------------------------------
  // A three-cycle: every vertex is symmetric, so every rank is exactly 1/3.
  // ---------------------------------------------------------------------
  {
    const Matrix adjacency = FromEdges(3, {{0, 1}, {1, 2}, {2, 0}});
    const Matrix op = spcraft::pagerank_operator(adjacency);

    Vector rank(3);
    const auto result = spcraft::pagerank_openmp(op, rank);
    if (!result.converged) {
      std::cerr << "PageRank on a 3-cycle failed to converge\n";
      ++failures;
    }
    for (Index vertex = 0; vertex < 3; ++vertex) {
      if (!Close(rank.val[vertex], 1.0 / 3.0)) {
        std::cerr << "PageRank on a 3-cycle: rank[" << vertex << "] = " << rank.val[vertex]
                  << ", expected 1/3\n";
        ++failures;
      }
    }
  }

  // ---------------------------------------------------------------------
  // Two vertices pointing at a third, which is a sink (dangling). Checked
  // against a closed form: with rank summing to one and 2 -> nothing,
  // r0 = r1 = (1-d)/3 + d/3 * r2 and r2 = (1-d)/3 + d*(r0+r1) + d/3*r2.
  // ---------------------------------------------------------------------
  {
    const Value damping = 0.85;
    const Matrix adjacency = FromEdges(3, {{0, 2}, {1, 2}});
    const Matrix op = spcraft::pagerank_operator(adjacency);
    if (op.nnz != 2) {
      std::cerr << "PageRank operator kept " << op.nnz << " entries, expected 2\n";
      ++failures;
    }

    Vector rank(3);
    spcraft::PageRankOptions<Value> options;
    options.damping = damping;
    options.tolerance = 1e-14;
    options.max_iterations = 500;
    const auto result = spcraft::pagerank_openmp(op, rank, options);
    if (!result.converged) {
      std::cerr << "PageRank with a dangling vertex failed to converge\n";
      ++failures;
    }

    // Solve the fixed point directly. r0 = r1 = a, r2 = 1 - 2a, and
    // a = (1-d)/3 + (d/3)(1 - 2a)  =>  a (1 + 2d/3) = (1-d)/3 + d/3.
    const Value a = ((1.0 - damping) / 3.0 + damping / 3.0) / (1.0 + 2.0 * damping / 3.0);
    const std::array<Value, 3> expected{a, a, 1.0 - 2.0 * a};
    for (Index vertex = 0; vertex < 3; ++vertex) {
      if (!Close(rank.val[vertex], expected[static_cast<std::size_t>(vertex)], 1e-10)) {
        std::cerr << "PageRank with a dangling vertex: rank[" << vertex
                  << "] = " << rank.val[vertex] << ", expected "
                  << expected[static_cast<std::size_t>(vertex)] << "\n";
        ++failures;
      }
    }
  }

  // ---------------------------------------------------------------------
  // An asymmetric graph cross-checked against the reference iteration, and
  // the ranks must sum to one.
  // ---------------------------------------------------------------------
  {
    const Index vertices = 6;
    const std::vector<std::pair<Index, Index>> edges{{0, 1}, {0, 2}, {1, 2}, {2, 0}, {3, 0},
                                                     {3, 1}, {3, 4}, {4, 5}, {5, 4}, {5, 0}};
    const Matrix adjacency = FromEdges(vertices, edges);
    const Matrix op = spcraft::pagerank_operator(adjacency);

    Vector rank(vertices);
    spcraft::PageRankOptions<Value> options;
    options.tolerance = 1e-15;
    options.max_iterations = 200;
    const auto result = spcraft::pagerank_openmp(op, rank, options);

    const auto expected = ReferencePageRank(vertices, edges, options.damping, 200);
    for (Index vertex = 0; vertex < vertices; ++vertex) {
      if (!Close(rank.val[vertex], expected[static_cast<std::size_t>(vertex)], 1e-9)) {
        std::cerr << "PageRank disagrees with the reference at vertex " << vertex << ": "
                  << rank.val[vertex] << " vs " << expected[static_cast<std::size_t>(vertex)]
                  << "\n";
        ++failures;
      }
    }

    const Value mass = std::accumulate(rank.val, rank.val + vertices, 0.0);
    if (!Close(mass, 1.0, 1e-12)) {
      std::cerr << "PageRank ranks sum to " << mass << ", expected 1\n";
      ++failures;
    }
    if (result.iterations > options.max_iterations) {
      std::cerr << "PageRank reported more iterations than the cap\n";
      ++failures;
    }
  }

  // ---------------------------------------------------------------------
  // A graph with no edges at all: every vertex dangling, ranks stay uniform.
  // ---------------------------------------------------------------------
  {
    Matrix adjacency;
    adjacency.Allocate(0, 4, 4);
    const Matrix op = spcraft::pagerank_operator(adjacency);
    if (op.nnz != 0) {
      std::cerr << "PageRank operator on an edgeless graph kept entries\n";
      ++failures;
    }

    Vector rank(4);
    const auto result = spcraft::pagerank_openmp(op, rank);
    if (!result.converged || result.iterations != 1) {
      std::cerr << "PageRank on an edgeless graph should converge in one iteration, took "
                << result.iterations << "\n";
      ++failures;
    }
    for (Index vertex = 0; vertex < 4; ++vertex) {
      if (!Close(rank.val[vertex], 0.25)) {
        std::cerr << "PageRank on an edgeless graph: rank[" << vertex
                  << "] = " << rank.val[vertex] << ", expected 0.25\n";
        ++failures;
      }
    }
  }

  // ---------------------------------------------------------------------
  // Rejected inputs.
  // ---------------------------------------------------------------------
  {
    Matrix rectangular;
    rectangular.Allocate(0, 2, 3);
    bool threw = false;
    try {
      (void)spcraft::pagerank_operator(rectangular);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) {
      std::cerr << "PageRank operator accepted a rectangular matrix\n";
      ++failures;
    }

    const Matrix op = spcraft::pagerank_operator(FromEdges(3, {{0, 1}, {1, 2}, {2, 0}}));
    Vector wrong_size(2);
    threw = false;
    try {
      (void)spcraft::pagerank_openmp(op, wrong_size);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) {
      std::cerr << "PageRank accepted a rank vector of the wrong size\n";
      ++failures;
    }

    Vector rank(3);
    spcraft::PageRankOptions<Value> options;
    options.damping = 1.0;
    threw = false;
    try {
      (void)spcraft::pagerank_openmp(op, rank, options);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) {
      std::cerr << "PageRank accepted a damping factor of 1\n";
      ++failures;
    }
  }

  if (failures != 0) {
    std::cerr << failures << " PageRank check(s) failed\n";
    return 1;
  }
  std::cout << "PageRank OpenMP checks passed\n";
  return 0;
}
