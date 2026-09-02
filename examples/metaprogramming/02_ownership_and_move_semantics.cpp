/**
 * @file 02_ownership_and_move_semantics.cpp
 * @brief Demonstrates Move Semantics, Resource Ownership (RAII), and Compile-Time Error Prevention.
 *
 * In high-performance sparse matrix computation, matrices often occupy gigabytes of memory.
 * Accidental copy construction (e.g. passing by value or implicit assignment) would cause:
 *  1. Catastrophic latency & memory bloat (accidental multi-GB deep copies).
 *  2. Pointer aliasing and double-free crashes if shallow-copied.
 *
 * SPCraft enforces compile-time safety by explicitly deleting the copy constructor and copy assignment:
 *   CsrMatrix(const CsrMatrix&) = delete;
 *   CsrMatrix& operator=(const CsrMatrix&) = delete;
 *
 * This example directly reproduces and explains the compiler error demonstrated in `e02_compiler_error.cpp`,
 * and shows how to properly use:
 *  - Move Semantics (`std::move`) for zero-cost ownership transfer.
 *  - Explicit `.Clone()` for intentional deep copies.
 *  - Non-owning views (`memowned = false`) to wrap external buffers without taking ownership.
 */

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "SpCraft.h"

// A function that consumes a matrix by taking ownership (move semantics)
template <typename IT, typename NT, typename OT>
void ProcessAndConsume(spcraft::CsrMatrix<IT, NT, OT> matrix)
{
  fmt::print("  [ProcessAndConsume] Received matrix with m={}, nnz={}, memowned={}\n",
             matrix.m, matrix.nnz, matrix.memowned);
  // Storage will be automatically and safely freed when `matrix` goes out of scope here!
}

// A function that inspects a matrix without taking ownership (read-only reference)
template <typename IT, typename NT, typename OT>
void InspectMatrix(const spcraft::CsrMatrix<IT, NT, OT>& matrix)
{
  fmt::print("  [InspectMatrix] Viewing matrix: m={}, n={}, nnz={}, memowned={}\n",
             matrix.m, matrix.n, matrix.nnz, matrix.memowned);
}

int main()
{
  fmt::print("============================================================\n");
  fmt::print(" Metaprogramming Pattern 2: Ownership, RAII & Move Semantics\n");
  fmt::print("============================================================\n");

  using CSR = spcraft::CsrMatrix<int32_t, double, int32_t>;

  // 1. Create and allocate an owning sparse matrix
  CSR A;
  A.Allocate(/*require_nnz=*/3, /*nRows=*/2, /*nCols=*/2);
  A.row_ptr[0] = 0; A.row_ptr[1] = 2; A.row_ptr[2] = 3;
  A.col_id[0] = 0;  A.col_id[1] = 1;  A.col_id[2] = 1;
  A.val[0] = 10.0;  A.val[1] = 20.0;  A.val[2] = 30.0;

  fmt::print("\n1. Created Initial Matrix A:\n");
  fmt::print("   A.row_ptr: {}\n", fmt::ptr(A.row_ptr));
  fmt::print("   A.memowned: {}\n", A.memowned);

  // --------------------------------------------------------------------------
  // 2. Why e02_compiler_error.cpp fails:
  // --------------------------------------------------------------------------
  // If you uncomment either of the following lines, the compiler halts with:
  // "error: call to deleted constructor of 'spcraft::CsrMatrix<...>'"
  //
  //   CSR B = A;          // COMPILE ERROR: Copy constructor is deleted!
  //   CSR C; C = A;       // COMPILE ERROR: Copy assignment is deleted!
  //
  // This compile-time check protects your application from silent, costly performance bugs!
  fmt::print("\n2. Compile-Time Safety Check:\n");
  fmt::print("   'CSR B = A;' is blocked at compile time by '= delete'.\n");

  // --------------------------------------------------------------------------
  // 3. Move Construction (Zero-cost transfer of pointer ownership)
  // --------------------------------------------------------------------------
  fmt::print("\n3. Move Semantics (Zero-cost ownership transfer):\n");
  int32_t* orig_row_ptr = A.row_ptr;
  CSR B = std::move(A);

  fmt::print("   After 'CSR B = std::move(A)':\n");
  fmt::print("   B.row_ptr: {} (takes ownership)\n", fmt::ptr(B.row_ptr));
  fmt::print("   A.row_ptr: {} (reset to nullptr)\n", fmt::ptr(A.row_ptr));
  fmt::print("   A.nnz: {} (reset to 0)\n", A.nnz);
  assert(B.row_ptr == orig_row_ptr);
  assert(A.row_ptr == nullptr);

  // --------------------------------------------------------------------------
  // 4. Explicit Deep Copy via Clone()
  // --------------------------------------------------------------------------
  fmt::print("\n4. Explicit Deep Copy via .Clone():\n");
  CSR C = B.Clone();
  fmt::print("   C.row_ptr: {} (newly allocated buffer)\n", fmt::ptr(C.row_ptr));
  fmt::print("   B.row_ptr: {} (original retained)\n", fmt::ptr(B.row_ptr));
  assert(C.row_ptr != B.row_ptr);
  assert(C.val[0] == B.val[0]);

  // --------------------------------------------------------------------------
  // 5. Non-Owning View Semantics (Wrapping external / C / Fortran buffers)
  // --------------------------------------------------------------------------
  fmt::print("\n5. Non-Owning View Semantics (memowned = false):\n");
  std::vector<int32_t> ext_row_ptr = {0, 1, 2};
  std::vector<int32_t> ext_col_id = {0, 1};
  std::vector<double>  ext_val = {100.0, 200.0};

  // Construct non-owning view
  CSR view(ext_row_ptr.data(), ext_col_id.data(), ext_val.data(),
           /*nnz=*/2, /*m=*/2, /*n=*/2);

  fmt::print("   view.memowned: {} (will NOT call delete[] on destruction)\n", view.memowned);
  InspectMatrix(view);

  // Transfer ownership to consuming function
  fmt::print("\n6. Consuming Matrix B:\n");
  ProcessAndConsume(std::move(B));
  assert(B.row_ptr == nullptr);

  fmt::print("\nOwnership and move semantics demonstration completed successfully!\n");
  return 0;
}
