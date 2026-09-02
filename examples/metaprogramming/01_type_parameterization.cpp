/**
 * @file 01_type_parameterization.cpp
 * @brief Demonstrates 3-Tier Generic Type Parameterization (IT, NT, OT) in SPCraft.
 *
 * In high-performance sparse matrix computation, type decoupling is essential:
 * 1. IT (Index Type, e.g. int32_t): Used for column indices and row/column dimension limits.
 * 2. NT (Numeric Type, e.g. float, double, std::complex): Arithmetic payload.
 * 3. OT (Offset Type, e.g. int64_t): Nonzero counts (nnz) and row pointer offsets.
 *
 * Decoupling OT from IT allows handling massive sparse matrices (e.g. nnz > 2^31)
 * without wasting 50% extra memory bandwidth by expanding all column indices to 64-bit.
 */

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "SpCraft.h"

// Helper function to print matrix storage footprint analysis
template <typename IT, typename NT, typename OT>
void AnalyzeMatrixMemory(const std::string& type_label, IT m, IT n, OT nnz)
{
  // Calculate buffer sizes in bytes
  size_t row_ptr_bytes = static_cast<size_t>(m + 1) * sizeof(OT);
  size_t col_id_bytes = static_cast<size_t>(nnz) * sizeof(IT);
  size_t val_bytes = static_cast<size_t>(nnz) * sizeof(NT);
  size_t total_bytes = row_ptr_bytes + col_id_bytes + val_bytes;

  fmt::print("\n=== Configuration: {} ===\n", type_label);
  fmt::print("  IT (Index):  {} ({} bytes)\n", typeid(IT).name(), sizeof(IT));
  fmt::print("  NT (Value):  {} ({} bytes)\n", typeid(NT).name(), sizeof(NT));
  fmt::print("  OT (Offset): {} ({} bytes)\n", typeid(OT).name(), sizeof(OT));
  fmt::print("  Dimensions:  {} x {}, Nonzeros (nnz): {}\n", m, n, nnz);
  fmt::print("  Memory Footprint:\n");
  fmt::print("    - row_ptr ({} elements): {:.2f} MB\n", m + 1,
             static_cast<double>(row_ptr_bytes) / (1024.0 * 1024.0));
  fmt::print("    - col_id  ({} elements): {:.2f} MB\n", nnz,
             static_cast<double>(col_id_bytes) / (1024.0 * 1024.0));
  fmt::print("    - val     ({} elements): {:.2f} MB\n", nnz,
             static_cast<double>(val_bytes) / (1024.0 * 1024.0));
  fmt::print("    - Total Storage:           {:.2f} MB ({:.2f} GB)\n",
             static_cast<double>(total_bytes) / (1024.0 * 1024.0),
             static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));
}

int main()
{
  fmt::print("============================================================\n");
  fmt::print(" Metaprogramming Pattern 1: 3-Tier Type Parameterization\n");
  fmt::print("============================================================\n");

  // 1. Standard 32-bit index and 32-bit offset configuration
  using SmallCSR = spcraft::CsrMatrix<int32_t, float, int32_t>;
  SmallCSR A;
  A.Allocate(/*nnz=*/4, /*nRows=*/3, /*nCols=*/3);

  A.row_ptr[0] = 0;
  A.row_ptr[1] = 2;
  A.row_ptr[2] = 3;
  A.row_ptr[3] = 4;

  A.col_id[0] = 0;
  A.col_id[1] = 2;
  A.col_id[2] = 1;
  A.col_id[3] = 2;

  A.val[0] = 1.5f;
  A.val[1] = 2.5f;
  A.val[2] = 3.5f;
  A.val[3] = 4.5f;

  fmt::print("SmallCSR successfully allocated: m={}, n={}, nnz={}\n", A.m, A.n, A.nnz);

  // 2. Large graph configuration: 32-bit indices (up to 2 billion vertices),
  //    64-bit offsets (over 4 billion edges), 64-bit double precision values.
  using LargeGraphCSR = spcraft::CsrMatrix<int32_t, double, int64_t>;

  // Simulate footprint analysis for a web graph matrix:
  // m = 50 million vertices, nnz = 3 billion edges
  int32_t num_vertices = 50'000'000;
  int64_t num_edges = 3'000'000'000LL;

  // Compare 32-bit index + 64-bit offset VS pure 64-bit index + 64-bit offset
  AnalyzeMatrixMemory<int32_t, double, int64_t>(
      "Hybrid Decoupled: IT=int32_t, NT=double, OT=int64_t (SPCraft Optimal)", num_vertices,
      num_vertices, num_edges);

  AnalyzeMatrixMemory<int64_t, double, int64_t>(
      "Monolithic 64-bit: IT=int64_t, NT=double, OT=int64_t (Excess Index Bandwidth)",
      num_vertices, num_vertices, num_edges);

  size_t hybrid_bytes = (static_cast<size_t>(num_vertices + 1) * sizeof(int64_t)) +
                        (static_cast<size_t>(num_edges) * sizeof(int32_t)) +
                        (static_cast<size_t>(num_edges) * sizeof(double));

  size_t monolithic_bytes = (static_cast<size_t>(num_vertices + 1) * sizeof(int64_t)) +
                            (static_cast<size_t>(num_edges) * sizeof(int64_t)) +
                            (static_cast<size_t>(num_edges) * sizeof(double));

  size_t saved_bytes = monolithic_bytes - hybrid_bytes;
  fmt::print(
      "\n>> Memory Bandwidth Saved by Template Type Decoupling: {:.2f} GB ({:.1f}% "
      "reduction)\n",
      static_cast<double>(saved_bytes) / (1024.0 * 1024.0 * 1024.0),
      (static_cast<double>(saved_bytes) / monolithic_bytes) * 100.0);

  return 0;
}
