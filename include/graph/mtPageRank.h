#pragma once

#include "utils/omp/omp_wrapper.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "core/CsrMatrix.h"
#include "core/DenseVector.h"
#include "graph/PageRank.h"
#include "kernel/mtSpMV.h"
#include "semiring/SemiRing.h"

namespace spcraft
{

/**
 * @brief PageRank by power iteration on top of the OpenMP SpMV kernel.
 *
 * Each iteration is one SpMV against the transposed operator plus two linear
 * passes:
 *
 *     y     = P^T . rank
 *     base  = ((1 - d) + d * (1 - sum(y))) / n
 *     rank' = base + d * y
 *
 * The `1 - sum(y)` term is the dangling mass. Columns of P^T belonging to
 * dangling vertices are empty, so exactly the rank those vertices held goes
 * missing from `sum(y)`; adding it back uniformly is equivalent to the usual
 * formulation with an explicit dangling-vertex indicator, without the extra
 * vector. Because the update is exactly stochastic, `rank` sums to one after
 * every iteration and no renormalization is needed.
 *
 * The iteration is defined over the reals, so unlike the SpMV kernels this
 * entry point is not templated on a semiring -- it fixes PlusTimesRing for the
 * multiply, which is the only ring for which PageRank means anything.
 *
 * @param op Transposed operator from pagerank_operator(); square, n x n.
 * @param rank Output ranks, size n. Overwritten with 1/n before iterating.
 * @param options Damping, tolerance, and iteration cap.
 * @return Iterations performed, final L1 residual, and whether it converged.
 * @throws std::invalid_argument on a non-square operator, a size mismatch, or
 *         options outside their valid ranges.
 */
template <class IT, class NT, class OT>
PageRankResult<NT> pagerank_openmp(const CsrMatrix<IT, NT, OT>& op, DenseVector<IT, NT>& rank,
                                   const PageRankOptions<NT>& options = PageRankOptions<NT>{})
{
  static_assert(std::is_floating_point_v<NT>, "PageRank requires a floating-point value type");

  if (op.m != op.n) {
    throw std::invalid_argument("PageRank requires a square operator");
  }
  if (rank.n != op.m) {
    throw std::invalid_argument("PageRank rank vector does not match the operator");
  }
  if (!detail::PageRankOptionsAreValid(options)) {
    throw std::invalid_argument("PageRank options are out of range");
  }

  const IT n = op.m;
  PageRankResult<NT> result;
  if (n == IT{0}) {
    result.converged = true;
    return result;
  }

  const NT damping = options.damping;
  const NT uniform = NT{1} / static_cast<NT>(n);

  OMP_PARALLEL_FOR(schedule(static) default(none) shared(rank, n, uniform))
  for (IT vertex = 0; vertex < n; ++vertex) {
    rank.val[vertex] = uniform;
  }

  DenseVector<IT, NT> y(n);

  for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
    OmpSpMV<PlusTimesRing<NT>>(op, rank, y);

    NT total = NT{0};
    OMP_PARALLEL_FOR(schedule(static) default(none) shared(y, n) reduction(+ : total))
    for (IT vertex = 0; vertex < n; ++vertex) {
      total += y.val[vertex];
    }

    // Teleport mass plus the mass the dangling vertices failed to hand on.
    const NT base = ((NT{1} - damping) + damping * (NT{1} - total)) * uniform;

    // rank is only read by the SpMV above, so the update runs in place and the
    // residual falls out of the same pass.
    NT residual = NT{0};
    OMP_PARALLEL_FOR(schedule(static) default(none) shared(rank, y, n, base, damping)
                         reduction(+ : residual))
    for (IT vertex = 0; vertex < n; ++vertex) {
      const NT updated = base + damping * y.val[vertex];
      residual += std::abs(updated - rank.val[vertex]);
      rank.val[vertex] = updated;
    }

    result.iterations = iteration;
    result.residual = residual;
    if (residual <= options.tolerance) {
      result.converged = true;
      break;
    }
  }

  return result;
}

}  // namespace spcraft
