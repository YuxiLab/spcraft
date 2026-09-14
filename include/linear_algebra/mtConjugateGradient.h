#pragma once

#include "utils/omp/omp_wrapper.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "core/CsrMatrix.h"
#include "core/DenseVector.h"
#include "kernel/mtSpMV.h"
#include "linear_algebra/ConjugateGradient.h"
#include "kernel/mtBlas1.h"
#include "semiring/SemiRing.h"

namespace spcraft
{

/**
 * @brief Solve A x = b for a symmetric positive definite A, using OpenMP SpMV.
 *
 * Unpreconditioned conjugate gradient. Each iteration is one SpMV against A
 * plus two dot products and three vector updates:
 *
 *     q     = A p
 *     alpha = rho / (p . q)
 *     x    += alpha p
 *     r    -= alpha q
 *     beta  = (r . r) / rho
 *     p     = r + beta p
 *
 * The SpMV is the only part that touches the matrix, so the cost per iteration
 * is the SpMV cost plus O(n) streaming work, and everything the CSR traversal
 * does for bandwidth applies here unchanged.
 *
 * A must be symmetric positive definite and stored with **all** its nonzeros,
 * not just one triangle -- the SpMV kernel reads rows verbatim and does not
 * reconstruct a mirrored half. Symmetry is not checked; supplying a
 * non-symmetric matrix silently solves a different problem. Indefiniteness is
 * caught, as a p^T A p that is not positive, and reported as a breakdown.
 *
 * Like PageRank this solve is not templated on a semiring: CG divides and
 * subtracts, so it needs a field rather than a semiring, and fixes
 * PlusTimesRing for the multiply.
 *
 * @param A Square SPD matrix in CSR, all nonzeros stored.
 * @param b Right-hand side, size n.
 * @param x On entry the initial guess, on exit the solution; size n. Zero it
 *          first unless you have a better starting point -- the routine does
 *          not clear it for you, so a warm start is one assignment away.
 * @param options Relative residual target and iteration cap.
 * @return Iterations performed, final relative residual, convergence, and
 *         whether the positive-definiteness test failed.
 * @throws std::invalid_argument on a non-square matrix, a size mismatch, or
 *         options outside their valid ranges.
 */
template <class IT, class NT, class OT>
CgResult<NT> cg_openmp(const CsrMatrix<IT, NT, OT>& A, const DenseVector<IT, NT>& b,
                       DenseVector<IT, NT>& x, const CgOptions<NT>& options = CgOptions<NT>{})
{
  static_assert(std::is_floating_point_v<NT>, "CG requires a floating-point value type");

  if (A.m != A.n) {
    throw std::invalid_argument("CG requires a square matrix");
  }
  if (b.n != A.m || x.n != A.m) {
    throw std::invalid_argument("CG vector dimensions do not match the matrix");
  }
  if (!detail::CgOptionsAreValid(options)) {
    throw std::invalid_argument("CG options are out of range");
  }

  const IT n = A.m;
  CgResult<NT> result;
  if (n == IT{0}) {
    result.converged = true;
    return result;
  }

  // A zero right-hand side has the zero solution, and the relative residual
  // test would divide by zero.
  const NT b_norm = std::sqrt(detail::Dot(b, b));
  if (b_norm == NT{0}) {
    detail::Fill(NT{0}, x);
    result.converged = true;
    return result;
  }

  DenseVector<IT, NT> r(n);
  DenseVector<IT, NT> p(n);
  DenseVector<IT, NT> q(n);

  // r = b - A x, honouring whatever initial guess the caller left in x.
  OmpSpMV<PlusTimesRing<NT>>(A, x, q);
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(r, b, q, p, n))
  for (IT index = 0; index < n; ++index) {
    const NT residual = b.val[index] - q.val[index];
    r.val[index] = residual;
    p.val[index] = residual;
  }

  NT rho = detail::Dot(r, r);
  result.residual = std::sqrt(rho) / b_norm;
  if (result.residual <= options.tolerance) {
    result.converged = true;
    return result;
  }

  for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
    OmpSpMV<PlusTimesRing<NT>>(A, p, q);

    // p^T A p is positive for every nonzero p exactly when A is positive
    // definite, so this is both the step length and the SPD check.
    const NT curvature = detail::Dot(p, q);
    if (!(curvature > NT{0})) {
      result.iterations = iteration - 1;
      result.breakdown = true;
      return result;
    }

    const NT alpha = rho / curvature;
    detail::Axpy(alpha, p, x);
    detail::Axpy(-alpha, q, r);

    const NT rho_next = detail::Dot(r, r);
    result.iterations = iteration;
    result.residual = std::sqrt(rho_next) / b_norm;
    if (result.residual <= options.tolerance) {
      result.converged = true;
      break;
    }

    detail::Aypx(rho_next / rho, r, p);
    rho = rho_next;
  }

  return result;
}

}  // namespace spcraft
