/**
 * @file 03_concepts_and_constraints.cpp
 * @brief Demonstrates C++20 Concepts, Type Traits, and Compile-Time Error Diagnostics.
 *
 * In legacy C++ (C++98/C++11/C++14), unconstrained templates produced notoriously unreadable
 * compiler errors spanning hundreds of lines when invalid types were passed.
 *
 * In modern C++ (C++20), we use Concepts and Constraints to:
 *  1. Formally specify syntactic and semantic requirements on template arguments.
 *  2. Turn cryptic substitution failures into single-line, human-readable compiler errors.
 *  3. Enable elegant compile-time function overloading and static dispatch.
 */

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <concepts>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "SpCraft.h"

namespace spcraft::meta
{

// 1. Define C++20 Concepts for Sparse Computation Types
template <typename T>
concept SparseIndex = std::integral<T> && (sizeof(T) >= 2);

template <typename T>
concept SparseOffset = std::integral<T> && (sizeof(T) >= sizeof(int32_t));

template <typename T>
concept SparseValue = std::floating_point<T> || std::integral<T>;

// 2. Define a Concept for a Sparse Matrix Container
template <typename M>
concept SparseMatrixType = requires(M m) {
  { m.m } -> std::convertible_to<std::size_t>;
  { m.n } -> std::convertible_to<std::size_t>;
  { m.nnz } -> std::convertible_to<std::size_t>;
  { m.val } -> std::same_as<typename M::value_type*>;
};

// 3. Compile-time validated Safe CSR Matrix Wrapper
template <SparseIndex IT, SparseValue NT, SparseOffset OT = IT>
struct ValidatedCsrMatrix
{
  using index_type = IT;
  using value_type = NT;
  using offset_type = OT;

  // Compile-time assertions with clear, pedagogical diagnostics
  static_assert(sizeof(OT) >= sizeof(IT),
                "[SPCraft Error] Offset type OT must be at least as wide as Index type IT to prevent overflow!");
  static_assert(!std::is_pointer_v<NT>,
                "[SPCraft Error] Numeric type NT cannot be a pointer!");
  static_assert(std::is_arithmetic_v<NT>,
                "[SPCraft Error] Numeric type NT must be an arithmetic type (e.g. float, double, int)!");

  IT m = 0;
  IT n = 0;
  OT nnz = 0;
  OT* row_ptr = nullptr;
  IT* col_id = nullptr;
  NT* val = nullptr;
};

// 4. Constrained SpMV algorithm requiring matching types
template <SparseIndex IT, SparseValue NT, SparseOffset OT>
requires (sizeof(OT) >= sizeof(IT))
void ConstrainedSpMV(const spcraft::CsrMatrix<IT, NT, OT>& A,
                     const NT* x,
                     NT* y)
{
  for (IT row = 0; row < A.m; ++row) {
    NT sum{};
    for (OT pos = A.row_ptr[row]; pos < A.row_ptr[row + 1]; ++pos) {
      sum += A.val[pos] * x[A.col_id[pos]];
    }
    y[row] = sum;
  }
}

// 5. Concept-based Overloading: Specialized fast-path for floating-point vs integer
template <SparseIndex IT, std::floating_point NT, SparseOffset OT>
void ExecuteSpecializedKernel(const spcraft::CsrMatrix<IT, NT, OT>& A)
{
  fmt::print("  -> Dispatching to [Floating-Point SIMD/FMA Optimized Kernel] (NT={})\n",
             typeid(NT).name());
}

template <SparseIndex IT, std::integral NT, SparseOffset OT>
void ExecuteSpecializedKernel(const spcraft::CsrMatrix<IT, NT, OT>& A)
{
  fmt::print("  -> Dispatching to [Exact Integer / Bitwise Graph Kernel] (NT={})\n",
             typeid(NT).name());
}

}  // namespace spcraft::meta

int main()
{
  fmt::print("============================================================\n");
  fmt::print(" Metaprogramming Pattern 3: C++20 Concepts & Type Constraints\n");
  fmt::print("============================================================\n");

  // A. Valid instantiation satisfying all concepts
  spcraft::meta::ValidatedCsrMatrix<int32_t, double, int64_t> valid_matrix;
  fmt::print("A. ValidatedCsrMatrix<int32_t, double, int64_t> successfully instantiated.\n");

  // B. Demonstration of Concept-based Overloading
  spcraft::CsrMatrix<int32_t, double, int64_t> float_csr;
  spcraft::CsrMatrix<int32_t, int64_t, int64_t> int_csr;

  fmt::print("\nB. Concept Overload Resolution:\n");
  spcraft::meta::ExecuteSpecializedKernel(float_csr);
  spcraft::meta::ExecuteSpecializedKernel(int_csr);

  // C. Execute Constrained SpMV
  spcraft::CsrMatrix<int32_t, float, int32_t> A;
  A.Allocate(/*nnz=*/2, /*nRows=*/2, /*nCols=*/2);
  A.row_ptr[0] = 0; A.row_ptr[1] = 1; A.row_ptr[2] = 2;
  A.col_id[0] = 0;  A.col_id[1] = 1;
  A.val[0] = 3.0f;  A.val[1] = 4.0f;

  std::vector<float> x = {2.0f, 5.0f};
  std::vector<float> y(2, 0.0f);

  spcraft::meta::ConstrainedSpMV(A, x.data(), y.data());

  fmt::print("\nC. Constrained SpMV Result: y = [{}]\n", fmt::join(y, ", "));

  // --------------------------------------------------------------------------
  // D. Compile-Time Diagnostics Demonstration:
  // --------------------------------------------------------------------------
  // If you try to instantiate with invalid types, modern C++ gives clean error messages:
  //
  // 1. Passing a string as value type:
  //    spcraft::meta::ValidatedCsrMatrix<int32_t, std::string, int32_t> invalid_val;
  //    -> Compiler message: "constraints not satisfied for class template 'ValidatedCsrMatrix'"
  //       "because 'std::string' does not satisfy 'SparseValue'"
  //
  // 2. Passing 64-bit index with 32-bit offset:
  //    spcraft::meta::ValidatedCsrMatrix<int64_t, double, int32_t> invalid_offset;
  //    -> static_assert failed: "[SPCraft Error] Offset type OT must be at least as wide as Index type IT!"

  fmt::print("\nC++20 Concepts demonstration completed successfully!\n");
  return 0;
}
