// PageRank on top of SpMV.
//
// PageRank asks: if a web surfer clicks links at random, and now and then gets
// bored and jumps to a page picked uniformly at random, what fraction of their
// time do they spend on each page? The answer is the stationary distribution of
// that random walk, and power iteration finds it by simulating the walk one
// step at a time until the distribution stops moving.
//
// One step is a sparse matrix-vector product, which is why this lives in a
// sparse library at all. This example builds a tiny graph by hand, runs the
// iteration, and prints the ranking.

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <cxxopts.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "SpCraft.h"

using Index = std::int32_t;
using Value = double;
using Offset = std::int64_t;
using Matrix = spcraft::CsrMatrix<Index, Value, Offset>;
using Vector = spcraft::DenseVector<Index, Value>;

int main(int argc, char** argv)
{
  cxxopts::Options command_line(argv[0], "Rank a small directed graph with SPCraft PageRank");
  // clang-format off
  command_line.add_options()
      ("d,damping", "Probability of following an out-link", cxxopts::value<Value>()->default_value("0.85"))
      ("t,tolerance", "L1 convergence tolerance", cxxopts::value<Value>()->default_value("1e-10"))
      ("m,max-iterations", "Maximum PageRank iterations", cxxopts::value<int>()->default_value("100"))
      ("h,help", "Show usage");
  // clang-format on
  const auto arguments = command_line.parse(argc, argv);
  if (arguments.count("help") != 0) {
    fmt::print("{}", command_line.help());
    return 0;
  }

  // A six-page miniature web. Edge (from, to) means "from links to to".
  // Page 3 is linked by nobody, page 2 is linked by everyone nearby.
  const Index pages = 6;
  const std::vector<std::pair<Index, Index>> links{{0, 1}, {0, 2}, {1, 2}, {2, 0}, {3, 0},
                                                   {3, 1}, {3, 4}, {4, 5}, {5, 4}, {5, 0}};

  // Step 1: put the link graph in CSR, one row per page listing its out-links.
  // This is the adjacency matrix A, with a 1 wherever a link exists.
  std::vector<Offset> row_ptr(static_cast<std::size_t>(pages) + 1, 0);
  for (const auto& [from, to] : links) {
    (void)to;
    ++row_ptr[static_cast<std::size_t>(from) + 1];
  }
  for (Index page = 0; page < pages; ++page) {
    row_ptr[static_cast<std::size_t>(page) + 1] += row_ptr[static_cast<std::size_t>(page)];
  }

  Matrix adjacency;
  adjacency.Allocate(static_cast<Offset>(links.size()), pages, pages);
  std::copy(row_ptr.begin(), row_ptr.end(), adjacency.row_ptr);

  std::vector<Offset> cursor(row_ptr.begin(), row_ptr.end() - 1);
  for (const auto& [from, to] : links) {
    const Offset destination = cursor[static_cast<std::size_t>(from)]++;
    adjacency.col_id[destination] = to;
    adjacency.val[destination] = 1.0;
  }

  // Step 2: turn the adjacency matrix into the operator the iteration needs.
  //
  // A rank update has to gather from a page's *in-links*, but a CSR row holds
  // *out-links*. So the operator is the transpose, with each entry divided by
  // the number of out-links of the page it came from -- the probability that a
  // surfer on that page follows this particular link. Building it once here
  // keeps every iteration down to a single SpMV.
  const Matrix operator_transposed = spcraft::pagerank_operator(adjacency);
  fmt::print("operator: {} rows, {} entries kept out of {} links\n", operator_transposed.m,
             operator_transposed.nnz, adjacency.nnz);

  // Step 3: iterate. Damping is the probability of following a link rather
  // than jumping to a random page; 0.85 is the usual starting point.
  spcraft::PageRankOptions<Value> options;
  options.damping = arguments["damping"].as<Value>();
  options.tolerance = arguments["tolerance"].as<Value>();
  options.max_iterations = arguments["max-iterations"].as<int>();

  Vector rank(pages);
  const auto rank_result = spcraft::pagerank_openmp(operator_transposed, rank, options);

  fmt::print("damping {}, tolerance {:.3e}\n", options.damping, options.tolerance);
  fmt::print("{} after {} iteration(s), final L1 change {:.3e}\n",
             rank_result.converged ? "converged" : "stopped at the iteration cap",
             rank_result.iterations, rank_result.residual);

  // Step 4: report. Ranks always sum to one, so they read as percentages.
  std::vector<Index> order(static_cast<std::size_t>(pages));
  std::iota(order.begin(), order.end(), Index{0});
  std::sort(order.begin(), order.end(),
            [&rank](Index left, Index right) { return rank.val[left] > rank.val[right]; });

  fmt::print("\nrank  page  score\n");
  for (std::size_t position = 0; position < order.size(); ++position) {
    fmt::print("{:>4}  {:>4}  {:.6f}\n", position + 1, order[position], rank.val[order[position]]);
  }
  fmt::print("\ntotal {:.6f}\n", std::accumulate(rank.val, rank.val + pages, 0.0));

  return 0;
}
