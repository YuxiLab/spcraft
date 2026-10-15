#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "SpCraft.h"

namespace spcraft::tests
{
namespace
{

using Index = std::int32_t;
using Value = double;
using Offset = std::int64_t;
using Matrix = spcraft::CsrMatrix<Index, Value, Offset>;
using Vector = spcraft::DenseVector<Index, Value>;

Matrix FromDense(const Eigen::MatrixXd& dense)
{
  const Index rows = static_cast<Index>(dense.rows());
  const Index columns = static_cast<Index>(dense.cols());
  std::vector<Offset> row_ptr(static_cast<std::size_t>(rows) + 1, 0);
  for (Index row = 0; row < rows; ++row) {
    Offset count = 0;
    for (Index column = 0; column < columns; ++column) {
      if (dense(row, column) != 0.0) ++count;
    }
    row_ptr[static_cast<std::size_t>(row) + 1] = row_ptr[static_cast<std::size_t>(row)] + count;
  }

  Matrix matrix;
  matrix.Allocate(row_ptr.back(), rows, columns);
  std::copy(row_ptr.begin(), row_ptr.end(), matrix.row_ptr);

  Offset position = 0;
  for (Index row = 0; row < rows; ++row) {
    for (Index column = 0; column < columns; ++column) {
      if (dense(row, column) != 0.0) {
        matrix.col_id[position] = column;
        matrix.val[position] = dense(row, column);
        ++position;
      }
    }
  }
  return matrix;
}

void Fill(Vector& vector, Value value) { std::fill(vector.val, vector.val + vector.n, value); }

//! Eigenvector direction is only defined up to sign.
bool SameDirection(const Vector& x, const Eigen::VectorXd& expected, Value tolerance)
{
  Value forward = 0.0;
  Value backward = 0.0;
  for (Index index = 0; index < x.n; ++index) {
    forward = std::max(forward, std::abs(x.val[index] - expected(index)));
    backward = std::max(backward, std::abs(x.val[index] + expected(index)));
  }
  return std::min(forward, backward) <= tolerance;
}

}  // namespace

TEST(PowerIteration, DiagonalDominantEigenpair)
{
  // ---------------------------------------------------------------------
  // A diagonal matrix: the answer is the largest entry and its unit vector.
  // ---------------------------------------------------------------------

  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(4, 4);
  dense.diagonal() << 1.0, 2.0, 3.0, 10.0;
  const Matrix A = FromDense(dense);

  Vector x(4);
  Fill(x, 1.0);
  const auto result = spcraft::power_iteration_openmp(A, x);

  if (!result.converged) {
    ADD_FAILURE() << "Power iteration on a diagonal matrix did not converge\n";
  }
  if (std::abs(result.eigenvalue - 10.0) > 1e-9) {
    ADD_FAILURE() << "Power iteration gave eigenvalue " << result.eigenvalue << ", expected 10\n";
  }
  if (std::abs(std::abs(x.val[3]) - 1.0) > 1e-9) {
    ADD_FAILURE() << "Power iteration eigenvector is not e4\n";
  }
}

TEST(PowerIteration, KnownTwoByTwoEigenpair)
{
  // ---------------------------------------------------------------------
  // [[2,1],[1,2]] has eigenvalues 3 and 1; the dominant vector is [1,1]/sqrt2.
  // ---------------------------------------------------------------------

  Eigen::MatrixXd dense(2, 2);
  dense << 2.0, 1.0, 1.0, 2.0;
  const Matrix A = FromDense(dense);

  Vector x(2);
  x.val[0] = 1.0;
  x.val[1] = 0.25;
  const auto result = spcraft::power_iteration_openmp(A, x);

  if (!result.converged || std::abs(result.eigenvalue - 3.0) > 1e-9) {
    ADD_FAILURE() << "Power iteration on the 2x2 gave " << result.eigenvalue << ", expected 3\n";
  }
  Eigen::VectorXd expected(2);
  expected << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
  if (!SameDirection(x, expected, 1e-8)) {
    ADD_FAILURE() << "Power iteration on the 2x2 returned the wrong direction\n";
  }
}

TEST(PowerIteration, NegativeDominantEigenvalue)
{
  // ---------------------------------------------------------------------
  // A negative dominant eigenvalue flips the iterate's sign every step. The
  // eigenvalue residual still converges; a change-in-x test would not.
  // ---------------------------------------------------------------------

  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(3, 3);
  dense.diagonal() << -5.0, 1.0, 2.0;
  const Matrix A = FromDense(dense);

  Vector x(3);
  Fill(x, 1.0);
  const auto result = spcraft::power_iteration_openmp(A, x);

  if (!result.converged || std::abs(result.eigenvalue + 5.0) > 1e-9) {
    ADD_FAILURE() << "Power iteration with a negative dominant eigenvalue gave " << result.eigenvalue
                  << ", expected -5\n";
  }
}

TEST(PowerIteration, SymmetricEigenReference)
{
  // ---------------------------------------------------------------------
  // A symmetric sparse matrix cross-checked against Eigen's eigensolver.
  // ---------------------------------------------------------------------

  const int n = 60;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  for (int row = 0; row < n; ++row) {
    dense(row, row) = 2.0 + 0.05 * static_cast<double>(row);
    if (row + 1 < n) {
      dense(row, row + 1) = -1.0;
      dense(row + 1, row) = -1.0;
    }
    if (row + 7 < n) {
      dense(row, row + 7) = 0.5;
      dense(row + 7, row) = 0.5;
    }
  }
  const Matrix A = FromDense(dense);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(dense);
  if (solver.info() != Eigen::Success) {
    FAIL() << "Eigen failed to factor the reference matrix";
  }
  int dominant = 0;
  for (int index = 1; index < n; ++index) {
    if (std::abs(solver.eigenvalues()(index)) > std::abs(solver.eigenvalues()(dominant))) {
      dominant = index;
    }
  }
  const double expected_value = solver.eigenvalues()(dominant);
  const Eigen::VectorXd expected_vector = solver.eigenvectors().col(dominant);

  Vector x(n);
  for (Index index = 0; index < n; ++index) {
    x.val[index] = 1.0 + 0.01 * static_cast<Value>(index);
  }
  spcraft::PowerIterationOptions<Value> options;
  options.tolerance = 1e-11;
  options.max_iterations = 20000;
  const auto result = spcraft::power_iteration_openmp(A, x, options);

  if (!result.converged) {
    ADD_FAILURE() << "Power iteration did not converge on the reference matrix, residual " << result.residual
                  << "\n";
  }
  if (std::abs(result.eigenvalue - expected_value) > 1e-8) {
    ADD_FAILURE() << "Power iteration gave " << result.eigenvalue << ", Eigen gives " << expected_value
                  << "\n";
  }
  if (!SameDirection(x, expected_vector, 1e-6)) {
    ADD_FAILURE() << "Power iteration eigenvector disagrees with Eigen\n";
  }

  // The returned vector must be a unit vector.
  Value norm = 0.0;
  for (Index index = 0; index < n; ++index) norm += x.val[index] * x.val[index];
  if (std::abs(std::sqrt(norm) - 1.0) > 1e-12) {
    ADD_FAILURE() << "Power iteration did not return a unit vector\n";
  }
}

TEST(PowerIteration, ZeroStartingVector)
{
  // ---------------------------------------------------------------------
  // A zero starting vector is seeded rather than dividing by zero.
  // ---------------------------------------------------------------------

  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(3, 3);
  dense.diagonal() << 1.0, 4.0, 2.0;
  const Matrix A = FromDense(dense);

  Vector x(3);
  Fill(x, 0.0);
  const auto result = spcraft::power_iteration_openmp(A, x);
  if (!result.converged || std::abs(result.eigenvalue - 4.0) > 1e-9) {
    ADD_FAILURE() << "Power iteration from a zero vector gave " << result.eigenvalue << ", expected 4\n";
  }
}

TEST(PowerIteration, ZeroMatrix)
{
  // ---------------------------------------------------------------------
  // The zero matrix: every vector is a null-space vector, eigenvalue 0.
  // ---------------------------------------------------------------------

  Matrix A;
  A.Allocate(0, 3, 3);
  std::fill(A.row_ptr, A.row_ptr + A.m + 1, Offset{0});
  Vector x(3);
  Fill(x, 1.0);
  const auto result = spcraft::power_iteration_openmp(A, x);
  if (!result.converged || result.eigenvalue != 0.0 || result.iterations != 1) {
    ADD_FAILURE() << "Power iteration on the zero matrix did not stop at eigenvalue 0\n";
  }
}

TEST(PowerIteration, EqualMagnitudeEigenvaluesDoNotConverge)
{
  // ---------------------------------------------------------------------
  // diag(1, -1) has two eigenvalues of equal magnitude, so the iteration
  // cannot converge and must stop at the cap rather than claim success.
  // ---------------------------------------------------------------------

  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(2, 2);
  dense.diagonal() << 1.0, -1.0;
  const Matrix A = FromDense(dense);

  Vector x(2);
  Fill(x, 1.0);
  spcraft::PowerIterationOptions<Value> options;
  options.max_iterations = 50;
  const auto result = spcraft::power_iteration_openmp(A, x, options);
  if (result.converged || result.iterations != 50) {
    ADD_FAILURE() << "Power iteration claimed convergence on a +/- eigenvalue pair\n";
  }
}

TEST(PowerIteration, RejectedInputs)
{
  // ---------------------------------------------------------------------
  // Rejected inputs.
  // ---------------------------------------------------------------------

  Matrix rectangular;
  rectangular.Allocate(0, 2, 3);
  Vector two(2);
  Fill(two, 1.0);

  bool threw = false;
  try {
    (void)spcraft::power_iteration_openmp(rectangular, two);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    ADD_FAILURE() << "Power iteration accepted a rectangular matrix\n";
  }

  const Matrix A = FromDense(Eigen::MatrixXd::Identity(2, 2));
  Vector wrong_size(3);
  Fill(wrong_size, 1.0);
  threw = false;
  try {
    (void)spcraft::power_iteration_openmp(A, wrong_size);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    ADD_FAILURE() << "Power iteration accepted a vector of the wrong size\n";
  }

  spcraft::PowerIterationOptions<Value> bad;
  bad.tolerance = -1.0;
  threw = false;
  try {
    (void)spcraft::power_iteration_openmp(A, two, bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    ADD_FAILURE() << "Power iteration accepted a negative tolerance\n";
  }
}

}  // namespace spcraft::tests
