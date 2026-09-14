#pragma once

#include "utils/omp/omp_wrapper.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "core/CsrMatrix.h"
#include "core/DenseVector.h"
#include "kernel/mtSpMV.h"
#include "linear_algebra/PowerIteration.h"
#include "kernel/mtBlas1.h"
#include "semiring/SemiRing.h"

namespace spcraft
{

/**
 * @brief Dominant eigenpair of A by power iteration, using OpenMP SpMV.
 *
 * Repeatedly applies A and renormalizes, so the iterate rotates towards the
 * eigenvector of largest magnitude eigenvalue:
 *
 *     y      = A x
 *     lambda = x . y                       (Rayleigh quotient, ||x|| = 1)
 *     x      = y / ||y||
 *
 * One SpMV and two vector passes per iteration, the same shape as PageRank --
 * which is exactly this iteration on a stochastic operator, with the shift that
 * makes the dominant eigenvalue one.
 *
 * Convergence is measured as ||A x - lambda x||, the eigenvalue residual, not
 * as the change in x. The two differ: for a negative dominant eigenvalue the
 * iterate flips sign every step and never settles, while the residual falls
 * normally. That residual could be had for free from
 * ||A x - lambda x||^2 = ||y||^2 - lambda^2, but near convergence those two
 * terms agree to nearly every digit, and the cancellation would put a floor of
 * about sqrt(eps)*|lambda| on the achievable tolerance. It is computed
 * directly instead, fused into the pass that renormalizes.
 *
 * Convergence needs the dominant eigenvalue to be real, simple, and strictly
 * larger in magnitude than the rest. Matrices that violate this -- a
 * +/-lambda pair, or a complex pair -- will not converge, and the iteration cap
 * is what stops them. Symmetry is not required, though for a nonsymmetric A the
 * Rayleigh quotient is only an eigenvalue estimate until the iterate settles.
 *
 * @param A Square matrix in CSR.
 * @param x On entry the starting vector, on exit the unit eigenvector; size n.
 *          Need not be normalized. If it is exactly zero, a uniform vector is
 *          substituted so the routine is usable with a zeroed vector.
 * @param options Residual target and iteration cap.
 * @return Iterations performed, the eigenvalue, the relative residual, and
 *         whether it converged.
 * @throws std::invalid_argument on a non-square matrix, a size mismatch, or
 *         options outside their valid ranges.
 */
template <class IT, class NT, class OT>
PowerIterationResult<NT> power_iteration_openmp(
    const CsrMatrix<IT, NT, OT>& A, DenseVector<IT, NT>& x,
    const PowerIterationOptions<NT>& options = PowerIterationOptions<NT>{})
{
  static_assert(std::is_floating_point_v<NT>,
                "Power iteration requires a floating-point type");

  if (A.m != A.n) {
    throw std::invalid_argument("Power iteration requires a square matrix");
  }
  if (x.n != A.m) {
    throw std::invalid_argument("Power iteration vector does not match the matrix");
  }
  if (!detail::PowerIterationOptionsAreValid(options)) {
    throw std::invalid_argument("Power iteration options are out of range");
  }

  const IT n = A.m;
  PowerIterationResult<NT> result;
  if (n == IT{0}) {
    result.converged = true;
    return result;
  }

  NT norm = std::sqrt(detail::Dot(x, x));
  if (norm == NT{0}) {
    detail::Fill(NT{1} / std::sqrt(static_cast<NT>(n)), x);
  } else {
    detail::Scale(NT{1} / norm, x);
  }

  DenseVector<IT, NT> y(n);

  for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
    OmpSpMV<PlusTimesRing<NT>>(A, x, y);

    NT rayleigh = NT{0};
    NT y_squared = NT{0};
    OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, y, n)
                         reduction(+ : rayleigh, y_squared))
    for (IT index = 0; index < n; ++index) {
      rayleigh += x.val[index] * y.val[index];
      y_squared += y.val[index] * y.val[index];
    }

    result.iterations = iteration;
    result.eigenvalue = rayleigh;

    // A x = 0 makes (0, x) an exact eigenpair: x lies in the null space and
    // there is nothing left to rotate towards.
    norm = std::sqrt(y_squared);
    if (norm == NT{0}) {
      result.eigenvalue = NT{0};
      result.residual = NT{0};
      result.converged = true;
      return result;
    }

    // One pass computes the residual against the old iterate and writes the
    // new one; the old x is read before it is overwritten at the same index.
    const NT scale = NT{1} / norm;
    NT residual_squared = NT{0};
    OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, y, n, rayleigh, scale)
                         reduction(+ : residual_squared))
    for (IT index = 0; index < n; ++index) {
      const NT difference = y.val[index] - rayleigh * x.val[index];
      residual_squared += difference * difference;
      x.val[index] = y.val[index] * scale;
    }

    const NT absolute = std::sqrt(residual_squared);
    // Fall back to an absolute test when the eigenvalue estimate is zero, which
    // a relative test cannot express.
    result.residual = rayleigh == NT{0} ? absolute : absolute / std::abs(rayleigh);
    if (result.residual <= options.tolerance) {
      result.converged = true;
      break;
    }
  }

  return result;
}

}  // namespace spcraft
