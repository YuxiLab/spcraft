#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "SpCraft.h"

namespace
{

using Index = std::int32_t;
using Value = double;
using Offset = std::int64_t;
using Matrix = spcraft::CsrMatrix<Index, Value, Offset>;
using Vector = spcraft::DenseVector<Index, Value>;

//! Build CSR from a dense matrix, keeping every structurally present entry.
Matrix FromDense(const Eigen::MatrixXd& dense)
{
  const Index n = static_cast<Index>(dense.rows());
  std::vector<Offset> row_ptr(static_cast<std::size_t>(n) + 1, 0);
  for (Index row = 0; row < n; ++row) {
    Offset count = 0;
    for (Index column = 0; column < static_cast<Index>(dense.cols()); ++column) {
      if (dense(row, column) != 0.0) ++count;
    }
    row_ptr[static_cast<std::size_t>(row) + 1] =
        row_ptr[static_cast<std::size_t>(row)] + count;
  }

  Matrix matrix;
  matrix.Allocate(row_ptr.back(), n, static_cast<Index>(dense.cols()));
  std::copy(row_ptr.begin(), row_ptr.end(), matrix.row_ptr);

  Offset position = 0;
  for (Index row = 0; row < n; ++row) {
    for (Index column = 0; column < static_cast<Index>(dense.cols()); ++column) {
      if (dense(row, column) != 0.0) {
        matrix.col_id[position] = column;
        matrix.val[position] = dense(row, column);
        ++position;
      }
    }
  }
  return matrix;
}

//! 2D 5-point Laplacian on a grid x grid mesh: the canonical SPD test problem.
Eigen::MatrixXd Laplacian2D(int grid)
{
  const int n = grid * grid;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  for (int row = 0; row < grid; ++row) {
    for (int column = 0; column < grid; ++column) {
      const int index = row * grid + column;
      dense(index, index) = 4.0;
      if (row > 0) dense(index, index - grid) = -1.0;
      if (row + 1 < grid) dense(index, index + grid) = -1.0;
      if (column > 0) dense(index, index - 1) = -1.0;
      if (column + 1 < grid) dense(index, index + 1) = -1.0;
    }
  }
  return dense;
}

void Fill(Vector& vector, Value value)
{
  std::fill(vector.val, vector.val + vector.n, value);
}

}  // namespace

int main()
{
  int failures = 0;

  // ---------------------------------------------------------------------
  // A hand-checked 2x2 system. A = [[4,1],[1,3]], b = [1,2] has the exact
  // solution x = [1/11, 7/11].
  // ---------------------------------------------------------------------
  {
    Eigen::MatrixXd dense(2, 2);
    dense << 4.0, 1.0, 1.0, 3.0;
    const Matrix A = FromDense(dense);

    Vector b(2);
    b.val[0] = 1.0;
    b.val[1] = 2.0;
    Vector x(2);
    Fill(x, 0.0);

    const auto result = spcraft::cg_openmp(A, b, x);
    if (!result.converged || result.breakdown) {
      std::cerr << "CG on the 2x2 system did not converge\n";
      ++failures;
    }
    if (std::abs(x.val[0] - 1.0 / 11.0) > 1e-12 || std::abs(x.val[1] - 7.0 / 11.0) > 1e-12) {
      std::cerr << "CG on the 2x2 system gave [" << x.val[0] << ", " << x.val[1]
                << "], expected [1/11, 7/11]\n";
      ++failures;
    }
    // CG terminates in at most n iterations in exact arithmetic.
    if (result.iterations > 2) {
      std::cerr << "CG on the 2x2 system took " << result.iterations
                << " iterations, expected at most 2\n";
      ++failures;
    }
  }

  // ---------------------------------------------------------------------
  // A scaled identity has one distinct eigenvalue, so CG must land in one
  // iteration regardless of the right-hand side.
  // ---------------------------------------------------------------------
  {
    const Matrix A = FromDense(Eigen::MatrixXd::Identity(8, 8) * 3.0);
    Vector b(8);
    for (Index index = 0; index < 8; ++index) b.val[index] = static_cast<Value>(index) - 3.0;
    Vector x(8);
    Fill(x, 0.0);

    const auto result = spcraft::cg_openmp(A, b, x);
    if (!result.converged || result.iterations != 1) {
      std::cerr << "CG on a scaled identity took " << result.iterations
                << " iterations, expected 1\n";
      ++failures;
    }
    for (Index index = 0; index < 8; ++index) {
      if (std::abs(x.val[index] - b.val[index] / 3.0) > 1e-14) {
        std::cerr << "CG on a scaled identity is wrong at " << index << "\n";
        ++failures;
      }
    }
  }

  // ---------------------------------------------------------------------
  // 2D Laplacian, cross-checked against a dense Eigen LDLT factorization.
  // ---------------------------------------------------------------------
  {
    const int grid = 12;
    const Eigen::MatrixXd dense = Laplacian2D(grid);
    const Index n = static_cast<Index>(dense.rows());
    const Matrix A = FromDense(dense);

    Eigen::VectorXd rhs(n);
    for (Index index = 0; index < n; ++index) {
      rhs(index) = std::sin(0.7 * static_cast<double>(index)) + 0.5;
    }
    const Eigen::VectorXd expected = dense.ldlt().solve(rhs);

    Vector b(n);
    std::copy(rhs.data(), rhs.data() + n, b.val);
    Vector x(n);
    Fill(x, 0.0);

    spcraft::CgOptions<Value> options;
    options.tolerance = 1e-12;
    options.max_iterations = 500;
    const auto result = spcraft::cg_openmp(A, b, x, options);

    if (!result.converged || result.breakdown) {
      std::cerr << "CG on the 2D Laplacian did not converge, residual " << result.residual
                << "\n";
      ++failures;
    }
    if (result.iterations > n) {
      std::cerr << "CG on the 2D Laplacian took " << result.iterations
                << " iterations, more than the matrix dimension " << n << "\n";
      ++failures;
    }
    Value worst = 0.0;
    for (Index index = 0; index < n; ++index) {
      worst = std::max(worst, std::abs(x.val[index] - expected(index)));
    }
    if (worst > 1e-9) {
      std::cerr << "CG disagrees with the dense LDLT solve by " << worst << "\n";
      ++failures;
    }

    // The reported residual must match a freshly computed one.
    Vector product(n);
    spcraft::OmpSpMV<spcraft::PlusTimesRing<Value>>(A, x, product);
    Value residual_norm = 0.0;
    Value rhs_norm = 0.0;
    for (Index index = 0; index < n; ++index) {
      const Value difference = b.val[index] - product.val[index];
      residual_norm += difference * difference;
      rhs_norm += b.val[index] * b.val[index];
    }
    const Value relative = std::sqrt(residual_norm) / std::sqrt(rhs_norm);
    if (std::abs(relative - result.residual) > 1e-8) {
      std::cerr << "CG reported residual " << result.residual << " but recomputation gives "
                << relative << "\n";
      ++failures;
    }
  }

  // ---------------------------------------------------------------------
  // A warm start at the exact solution must be recognized without iterating.
  // ---------------------------------------------------------------------
  {
    Eigen::MatrixXd dense(2, 2);
    dense << 4.0, 1.0, 1.0, 3.0;
    const Matrix A = FromDense(dense);
    Vector b(2);
    b.val[0] = 1.0;
    b.val[1] = 2.0;
    Vector x(2);
    x.val[0] = 1.0 / 11.0;
    x.val[1] = 7.0 / 11.0;

    const auto result = spcraft::cg_openmp(A, b, x);
    if (!result.converged || result.iterations != 0) {
      std::cerr << "CG started at the solution took " << result.iterations
                << " iterations, expected 0\n";
      ++failures;
    }
  }

  // ---------------------------------------------------------------------
  // A zero right-hand side yields the zero solution, whatever x held before.
  // ---------------------------------------------------------------------
  {
    const Matrix A = FromDense(Eigen::MatrixXd::Identity(4, 4) * 2.0);
    Vector b(4);
    Fill(b, 0.0);
    Vector x(4);
    Fill(x, 17.0);

    const auto result = spcraft::cg_openmp(A, b, x);
    if (!result.converged || result.iterations != 0) {
      std::cerr << "CG with a zero right-hand side did not return immediately\n";
      ++failures;
    }
    for (Index index = 0; index < 4; ++index) {
      if (x.val[index] != 0.0) {
        std::cerr << "CG with a zero right-hand side left x nonzero\n";
        ++failures;
        break;
      }
    }
  }

  // ---------------------------------------------------------------------
  // An indefinite matrix must be reported, not silently iterated.
  // ---------------------------------------------------------------------
  {
    Eigen::MatrixXd dense(2, 2);
    dense << 1.0, 0.0, 0.0, -1.0;
    const Matrix A = FromDense(dense);
    Vector b(2);
    b.val[0] = 1.0;
    b.val[1] = 1.0;
    Vector x(2);
    Fill(x, 0.0);

    const auto result = spcraft::cg_openmp(A, b, x);
    if (!result.breakdown || result.converged) {
      std::cerr << "CG did not report a breakdown on an indefinite matrix\n";
      ++failures;
    }
  }

  // ---------------------------------------------------------------------
  // Rejected inputs.
  // ---------------------------------------------------------------------
  {
    Matrix rectangular;
    rectangular.Allocate(0, 2, 3);
    Vector two(2);
    Fill(two, 0.0);
    Vector other(2);
    Fill(other, 0.0);

    bool threw = false;
    try {
      (void)spcraft::cg_openmp(rectangular, two, other);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) {
      std::cerr << "CG accepted a rectangular matrix\n";
      ++failures;
    }

    const Matrix A = FromDense(Eigen::MatrixXd::Identity(2, 2));
    Vector wrong_size(3);
    Fill(wrong_size, 0.0);
    threw = false;
    try {
      (void)spcraft::cg_openmp(A, wrong_size, other);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) {
      std::cerr << "CG accepted a right-hand side of the wrong size\n";
      ++failures;
    }

    spcraft::CgOptions<Value> bad;
    bad.max_iterations = -1;
    threw = false;
    try {
      (void)spcraft::cg_openmp(A, two, other, bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) {
      std::cerr << "CG accepted a negative iteration cap\n";
      ++failures;
    }
  }

  if (failures != 0) {
    std::cerr << failures << " CG check(s) failed\n";
    return 1;
  }
  std::cout << "CG OpenMP checks passed\n";
  return 0;
}
