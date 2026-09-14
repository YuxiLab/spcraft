// Conjugate gradient on top of SpMV.
//
// This example discretizes the negative Laplacian on a square grid. The
// resulting five-point stencil is symmetric positive definite (SPD), exactly
// the matrix class conjugate gradient is designed for. We choose a smooth exact
// solution, form b = A x_exact with SPCraft SpMV, solve A x = b, and compare the
// recovered vector with the known answer.

#include <fmt/core.h>
#include <cxxopts.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

//! Build the full five-point stencil; CG needs both halves of the symmetric matrix.
Matrix Laplacian2D(Index grid)
{
  const std::int64_t point_count = static_cast<std::int64_t>(grid) * grid;
  if (point_count > std::numeric_limits<Index>::max()) {
    throw std::overflow_error("grid contains more points than the index type can represent");
  }
  const Index n = static_cast<Index>(point_count);

  std::vector<Offset> row_offsets(static_cast<std::size_t>(n) + 1, Offset{0});
  for (Index row = 0; row < grid; ++row) {
    for (Index column = 0; column < grid; ++column) {
      const Index point = row * grid + column;
      Offset entries = 1;  // diagonal
      if (row > 0) ++entries;
      if (row + 1 < grid) ++entries;
      if (column > 0) ++entries;
      if (column + 1 < grid) ++entries;
      row_offsets[static_cast<std::size_t>(point) + 1] =
          row_offsets[static_cast<std::size_t>(point)] + entries;
    }
  }

  Matrix matrix;
  matrix.Allocate(row_offsets.back(), n, n);
  std::copy(row_offsets.begin(), row_offsets.end(), matrix.row_ptr);

  for (Index row = 0; row < grid; ++row) {
    for (Index column = 0; column < grid; ++column) {
      const Index point = row * grid + column;
      Offset position = matrix.row_ptr[point];
      if (row > 0) {
        matrix.col_id[position] = point - grid;
        matrix.val[position++] = -1.0;
      }
      if (column > 0) {
        matrix.col_id[position] = point - 1;
        matrix.val[position++] = -1.0;
      }
      matrix.col_id[position] = point;
      matrix.val[position++] = 4.0;
      if (column + 1 < grid) {
        matrix.col_id[position] = point + 1;
        matrix.val[position++] = -1.0;
      }
      if (row + 1 < grid) {
        matrix.col_id[position] = point + grid;
        matrix.val[position] = -1.0;
      }
    }
  }
  return matrix;
}

}  // namespace

int main(int argc, char** argv)
{
  cxxopts::Options command_line(argv[0], "Solve a 2D Laplacian system with SPCraft CG");
  // clang-format off
  command_line.add_options()
      ("g,grid", "Grid points along one dimension", cxxopts::value<Index>()->default_value("32"))
      ("t,tolerance", "Relative residual tolerance", cxxopts::value<Value>()->default_value("1e-10"))
      ("m,max-iterations", "Maximum CG iterations", cxxopts::value<int>()->default_value("1000"))
      ("h,help", "Show usage");
  // clang-format on
  const auto arguments = command_line.parse(argc, argv);
  if (arguments.count("help") != 0) {
    fmt::print("{}", command_line.help());
    return 0;
  }

  const Index grid = arguments["grid"].as<Index>();
  if (grid <= 0) {
    fmt::print(stderr, "Error: --grid must be positive.\n");
    return 1;
  }

  const Matrix matrix = Laplacian2D(grid);
  Vector exact(matrix.n);
  Vector rhs(matrix.n);
  Vector solution(matrix.n);

  // This smooth field is zero on the boundary but is not a single eigenmode,
  // so the example exercises a genuine multi-iteration Krylov solve.
  const Value spacing = Value{1} / static_cast<Value>(grid + 1);
  for (Index row = 0; row < grid; ++row) {
    for (Index column = 0; column < grid; ++column) {
      const Value vertical = spacing * static_cast<Value>(row + 1);
      const Value horizontal = spacing * static_cast<Value>(column + 1);
      exact.val[row * grid + column] =
          horizontal * (Value{1} - horizontal) * vertical * (Value{1} - vertical) *
          std::exp(Value{0.5} * horizontal + Value{0.25} * vertical);
    }
  }
  spcraft::OmpSpMV<spcraft::PlusTimesRing<Value>>(matrix, exact, rhs);
  std::fill(solution.val, solution.val + solution.n, Value{0});

  spcraft::CgOptions<Value> options;
  options.tolerance = arguments["tolerance"].as<Value>();
  options.max_iterations = arguments["max-iterations"].as<int>();
  const auto result = spcraft::cg_openmp(matrix, rhs, solution, options);

  Value squared_error = 0.0;
  Value squared_exact = 0.0;
  Value largest_error = 0.0;
  for (Index point = 0; point < matrix.n; ++point) {
    const Value error = solution.val[point] - exact.val[point];
    squared_error += error * error;
    squared_exact += exact.val[point] * exact.val[point];
    largest_error = std::max(largest_error, std::abs(error));
  }

  fmt::print("matrix: {} x {}, {} nonzeros\n", matrix.m, matrix.n, matrix.nnz);
  fmt::print("status: {}{}\n", result.converged ? "converged" : "not converged",
             result.breakdown ? " (SPD breakdown)" : "");
  fmt::print("iterations: {}, relative residual: {:.3e}\n", result.iterations,
             result.residual);
  fmt::print("relative solution error: {:.3e}, largest entry error: {:.3e}\n",
             std::sqrt(squared_error / squared_exact), largest_error);

  return result.converged && !result.breakdown ? 0 : 1;
}
