#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
  if (argc != 2) {
    fmt::print(stderr, "Usage: {} <matrix-file.mtx>\n", argv[0]);
    return 1;
  }

  const std::string matrix_path = argv[1];
  std::ifstream input(matrix_path);
  if (!input.is_open()) {
    fmt::print(stderr, "Error: unable to open matrix file: {}\n", matrix_path);
    return 1;
  }

  // fast_matrix_market stores a sparse matrix as coordinate triplets:
  // (row_indices[i], column_indices[i], values[i]). Indices are zero-based.
  using Index = std::int64_t;
  using Value = double;

  std::int64_t rows = 0;
  std::int64_t columns = 0;
  std::vector<Index> row_indices;
  std::vector<Index> column_indices;
  std::vector<Value> values;

  try {
    fast_matrix_market::read_matrix_market_triplet(
        input, rows, columns, row_indices, column_indices, values);
  } catch (const std::exception& error) {
    fmt::print(stderr, "Error reading {}: {}\n", matrix_path, error.what());
    return 1;
  }

  fmt::print("Matrix: {}\n", matrix_path);
  fmt::print("Rows: {}\n", rows);
  fmt::print("Columns: {}\n", columns);
  fmt::print("Nonzeros: {}\n", values.size());

  const std::size_t entries_to_print = std::min<std::size_t>(values.size(), 10);
  fmt::print("\nFirst {} triplets (zero-based):\n", entries_to_print);
  for (std::size_t i = 0; i < entries_to_print; ++i) {
    fmt::print("  ({}, {}) = {}\n", row_indices[i], column_indices[i], values[i]);
  }

  return 0;
}
