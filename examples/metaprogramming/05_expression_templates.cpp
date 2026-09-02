/**
 * @file 05_expression_templates.cpp
 * @brief Demonstrates Expression Templates & Lazy Kernel Fusion for Sparse Linear Algebra.
 *
 * In high-performance linear algebra, naive operator overloading is notoriously slow:
 *   y = alpha * (A * x) + beta * z;
 *
 * Naive evaluation causes:
 *   1. Multiple temporary heap allocations (e.g. for `A * x`, `alpha * (...)`, `beta * z`).
 *   2. Multiple passes over memory (polluting CPU L1/L2 caches and saturating DRAM bandwidth).
 *
 * Expression Templates solve this by building a compile-time expression tree (AST).
 * Computation only executes upon assignment (`operator=`), fusing all scalar scaling and additions
 * into a single, streaming kernel loop with ZERO intermediate memory allocations.
 */

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "SpCraft.h"

namespace spcraft::meta
{

// 1. Base Expression Interface (CRTP: Curiously Recurring Template Pattern)
template <typename Derived>
struct VectorExpression {
  const Derived& derived() const { return static_cast<const Derived&>(*this); }
  auto operator[](std::size_t i) const { return derived()[i]; }
  std::size_t size() const { return derived().size(); }
};

// 2. Vector Wrapper Expression
template <typename T>
class VectorWrapper : public VectorExpression<VectorWrapper<T>> {
  const std::vector<T>& vec_;
 public:
  explicit VectorWrapper(const std::vector<T>& vec) : vec_(vec) {}
  T operator[](std::size_t i) const { return vec_[i]; }
  std::size_t size() const { return vec_.size(); }
};

// 3. Scalar Multiplication Expression: alpha * Expr
template <typename Expr, typename Scalar>
class ScaledExpression : public VectorExpression<ScaledExpression<Expr, Scalar>> {
  Expr expr_;
  Scalar alpha_;
 public:
  ScaledExpression(const Expr& expr, Scalar alpha) : expr_(expr), alpha_(alpha) {}
  auto operator[](std::size_t i) const { return alpha_ * expr_[i]; }
  std::size_t size() const { return expr_.size(); }
};

// 4. Vector Addition Expression: Expr1 + Expr2
template <typename LeftExpr, typename RightExpr>
class AddExpression : public VectorExpression<AddExpression<LeftExpr, RightExpr>> {
  LeftExpr left_;
  RightExpr right_;
 public:
  AddExpression(const LeftExpr& left, const RightExpr& right)
      : left_(left), right_(right) {
    assert(left.size() == right.size());
  }
  auto operator[](std::size_t i) const { return left_[i] + right_[i]; }
  std::size_t size() const { return left_.size(); }
};

// 5. Sparse Matrix-Vector Product Lazy Expression: (A * x)
template <typename IT, typename NT, typename OT>
class SpMVExpression : public VectorExpression<SpMVExpression<IT, NT, OT>> {
  const spcraft::CsrMatrix<IT, NT, OT>& A_;
  const std::vector<NT>& x_;
 public:
  SpMVExpression(const spcraft::CsrMatrix<IT, NT, OT>& A, const std::vector<NT>& x)
      : A_(A), x_(x) {
    assert(static_cast<std::size_t>(A.n) == x.size());
  }

  NT operator[](std::size_t row) const {
    NT sum{};
    for (OT pos = A_.row_ptr[row]; pos < A_.row_ptr[row + 1]; ++pos) {
      sum += A_.val[pos] * x_[A_.col_id[pos]];
    }
    return sum;
  }

  std::size_t size() const { return static_cast<std::size_t>(A_.m); }
};

// Overload Operators to build the compile-time AST
template <typename Expr, typename Scalar>
auto operator*(Scalar alpha, const VectorExpression<Expr>& expr) {
  return ScaledExpression<Expr, Scalar>(expr.derived(), alpha);
}

template <typename Left, typename Right>
auto operator+(const VectorExpression<Left>& left, const VectorExpression<Right>& right) {
  return AddExpression<Left, Right>(left.derived(), right.derived().derived());
}

template <typename IT, typename NT, typename OT>
auto operator*(const spcraft::CsrMatrix<IT, NT, OT>& A, const std::vector<NT>& x) {
  return SpMVExpression<IT, NT, OT>(A, x);
}

// Destination Vector that evaluates the compile-time expression in a single pass
template <typename T>
class CustomVector {
 public:
  std::vector<T> data;

  explicit CustomVector(std::size_t size, T initial_val = T{})
      : data(size, initial_val) {}

  // Fused Assignment Operator: Computes and stores result directly!
  template <typename Expr>
  CustomVector& operator=(const VectorExpression<Expr>& expr) {
    std::size_t n = expr.size();
    if (data.size() != n) {
      data.resize(n);
    }
    // Single loop kernel fusion: zero temporary allocations!
    for (std::size_t i = 0; i < n; ++i) {
      data[i] = expr[i];
    }
    return *this;
  }
};

}  // namespace spcraft::meta

using namespace spcraft;
using namespace spcraft::meta;

int main()
{
  fmt::print("============================================================\n");
  fmt::print(" Metaprogramming Pattern 5: Expression Templates & Fusion\n");
  fmt::print("============================================================\n");

  using CSR = spcraft::CsrMatrix<int32_t, double, int32_t>;

  // Construct matrix A:
  // [ 2.0  0.0 ]
  // [ 1.0  3.0 ]
  CSR A;
  A.Allocate(/*nnz=*/3, /*nRows=*/2, /*nCols=*/2);
  A.row_ptr[0] = 0; A.row_ptr[1] = 1; A.row_ptr[2] = 3;
  A.col_id[0] = 0;
  A.col_id[1] = 0;  A.col_id[2] = 1;
  A.val[0] = 2.0;
  A.val[1] = 1.0;   A.val[2] = 3.0;

  std::vector<double> x = {2.0, 4.0};
  std::vector<double> z = {10.0, 20.0};
  spcraft::meta::VectorWrapper<double> wrapped_z(z);

  double alpha = 2.5;
  double beta = 0.5;

  // Build the Expression Tree for: y = alpha * (A * x) + beta * z
  // Notice that NO vector allocation happens here!
  auto expr = (alpha * (A * x)) + (beta * wrapped_z);

  fmt::print("Expression Tree Type constructed at compile time:\n  {}\n\n",
             typeid(decltype(expr)).name());

  // Evaluate the fused expression in a single pass
  spcraft::meta::CustomVector<double> y(2);
  y = expr;

  // Expected computation:
  // A * x = [ 2*2 = 4, 1*2 + 3*4 = 14 ]
  // alpha * (A * x) = [ 2.5 * 4 = 10, 2.5 * 14 = 35 ]
  // beta * z = [ 0.5 * 10 = 5, 0.5 * 20 = 10 ]
  // y = [ 10 + 5 = 15, 35 + 10 = 45 ]

  fmt::print("Evaluated Fused Expression Result:\n");
  fmt::print("  y = [{}]\n", fmt::join(y.data, ", "));
  assert(y.data[0] == 15.0);
  assert(y.data[1] == 45.0);

  fmt::print("\nExpression templates and kernel fusion demonstrated successfully!\n");
  return 0;
}
