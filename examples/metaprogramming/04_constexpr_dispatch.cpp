/**
 * @file 04_constexpr_dispatch.cpp
 * @brief Demonstrates Compile-Time Branch Elimination with `if constexpr` and Static Dispatch.
 *
 * In numerical HPC, putting runtime `if (condition)` branches inside innermost loops
 * causes branch mispredictions, inhibits vectorization (SIMD), and cripples performance.
 *
 * With C++17/C++20 `if constexpr`, conditions evaluated on compile-time types or template constants
 * are completely eliminated by the compiler before code generation. Only the active branch is emitted.
 */

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include "SpCraft.h"

namespace spcraft::meta
{

enum class MatrixSymmetry {
  General,    // Store all nonzeros
  Symmetric   // Store only upper triangle (val[i,j] implies val[j,i])
};

/**
 * @brief Zero-overhead SpMV with compile-time symmetry branching.
 *
 * Notice: The `if constexpr (Symmetry == MatrixSymmetry::Symmetric)` branch
 * is evaluated entirely by the compiler during template instantiation!
 * The generated assembly for `General` will contain zero symmetry overhead.
 */
template <MatrixSymmetry Symmetry = MatrixSymmetry::General,
          typename IT, typename NT, typename OT>
void FastSpMV(const spcraft::CsrMatrix<IT, NT, OT>& A,
              const NT* x,
              NT* y)
{
  if constexpr (Symmetry == MatrixSymmetry::General) {
    // Branch A: Standard CSR SpMV loop (easily vectorized)
    for (IT row = 0; row < A.m; ++row) {
      NT sum{};
      for (OT pos = A.row_ptr[row]; pos < A.row_ptr[row + 1]; ++pos) {
        sum += A.val[pos] * x[A.col_id[pos]];
      }
      y[row] += sum;
    }
  } else if constexpr (Symmetry == MatrixSymmetry::Symmetric) {
    // Branch B: Symmetric SpMV (upper triangle update both y[i] and y[j])
    for (IT row = 0; row < A.m; ++row) {
      for (OT pos = A.row_ptr[row]; pos < A.row_ptr[row + 1]; ++pos) {
        IT col = A.col_id[pos];
        NT val = A.val[pos];
        y[row] += val * x[col];
        if (row != col) {
          y[col] += val * x[row];
        }
      }
    }
  }
}

/**
 * @brief Compile-time precision accumulator selection.
 * When NT is float (FP32), accumulates in double (FP64) to prevent precision loss.
 */
template <typename IT, typename NT, typename OT>
void HighPrecisionSpMV(const spcraft::CsrMatrix<IT, NT, OT>& A,
                       const NT* x,
                       NT* y)
{
  // Select accumulator type at compile time
  using AccumulatorType = std::conditional_t<std::is_same_v<NT, float>, double, NT>;

  for (IT row = 0; row < A.m; ++row) {
    AccumulatorType sum = 0.0;
    for (OT pos = A.row_ptr[row]; pos < A.row_ptr[row + 1]; ++pos) {
      sum += static_cast<AccumulatorType>(A.val[pos]) * static_cast<AccumulatorType>(x[A.col_id[pos]]);
    }
    y[row] = static_cast<NT>(sum);
  }
}

}  // namespace spcraft::meta

int main()
{
  fmt::print("============================================================\n");
  fmt::print(" Metaprogramming Pattern 4: Compile-Time `if constexpr` Dispatch\n");
  fmt::print("============================================================\n");

  using CSR = spcraft::CsrMatrix<int32_t, double, int32_t>;

  // Construct a 3x3 symmetric matrix (only upper triangular part stored)
  // [ 2.0  1.0  0.0 ]
  // [ 1.0  3.0  4.0 ]
  // [ 0.0  4.0  5.0 ]
  CSR upper_sym;
  upper_sym.Allocate(/*nnz=*/5, /*nRows=*/3, /*nCols=*/3);
  upper_sym.row_ptr[0] = 0; upper_sym.row_ptr[1] = 2; upper_sym.row_ptr[2] = 4; upper_sym.row_ptr[3] = 5;
  upper_sym.col_id[0] = 0; upper_sym.col_id[1] = 1; // row 0: (0,0)=2.0, (0,1)=1.0
  upper_sym.col_id[2] = 1; upper_sym.col_id[3] = 2; // row 1: (1,1)=3.0, (1,2)=4.0
  upper_sym.col_id[4] = 2;                          // row 2: (2,2)=5.0
  upper_sym.val[0] = 2.0; upper_sym.val[1] = 1.0;
  upper_sym.val[2] = 3.0; upper_sym.val[3] = 4.0;
  upper_sym.val[4] = 5.0;

  std::vector<double> x = {1.0, 2.0, 3.0};
  std::vector<double> y_sym(3, 0.0);

  // 1. Dispatch to Symmetric Kernel with zero runtime branch overhead
  spcraft::meta::FastSpMV<spcraft::meta::MatrixSymmetry::Symmetric>(upper_sym, x.data(), y_sym.data());
  fmt::print("1. Symmetric SpMV Result: y = [{}]\n", fmt::join(y_sym, ", "));
  // Expected:
  // y[0] = 2*1 + 1*2 = 4
  // y[1] = 1*1 + 3*2 + 4*3 = 19
  // y[2] = 4*2 + 5*3 = 23

  // 2. High precision accumulator test with FP32 values
  spcraft::CsrMatrix<int32_t, float, int32_t> float_mat;
  float_mat.Allocate(/*nnz=*/2, /*nRows=*/1, /*nCols=*/2);
  float_mat.row_ptr[0] = 0; float_mat.row_ptr[1] = 2;
  float_mat.col_id[0] = 0; float_mat.col_id[1] = 1;
  float_mat.val[0] = 1e8f; float_mat.val[1] = 1.0f;

  std::vector<float> x_fp32 = {1.0f, 1.0f};
  std::vector<float> y_fp32(1, 0.0f);

  spcraft::meta::HighPrecisionSpMV(float_mat, x_fp32.data(), y_fp32.data());
  fmt::print("2. High Precision SpMV (FP32 in FP64 accumulator): y[0] = {}\n", y_fp32[0]);

  fmt::print("\nCompile-time `if constexpr` dispatch demonstration completed successfully!\n");
  return 0;
}
