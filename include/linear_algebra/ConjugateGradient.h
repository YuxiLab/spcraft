#pragma once

namespace spcraft
{

/**
 * @brief Stopping controls for a Krylov solve.
 */
template <class NT>
struct CgOptions {
  NT tolerance = NT{1e-10};   //!< Relative residual target, ||r|| / ||b||.
  int max_iterations = 1000;  //!< Upper bound on iterations.
};

/**
 * @brief What a Krylov solve did.
 */
template <class NT>
struct CgResult {
  int iterations = 0;      //!< Iterations actually performed.
  NT residual = NT{0};     //!< Final relative residual, ||r|| / ||b||.
  bool converged = false;  //!< Whether the residual reached the tolerance.
  /**
   * @brief Set when a search direction failed the positive-definiteness test.
   *
   * CG is only defined for symmetric positive definite matrices. If p^T A p is
   * not positive the recurrence has no valid step length, which means either
   * the matrix is indefinite or rounding has destroyed the Krylov basis. The
   * solve stops and leaves the last usable iterate in x.
   */
  bool breakdown = false;
};

namespace detail
{

//! Reject option sets no solve can act on.
template <class NT>
[[nodiscard]] constexpr bool CgOptionsAreValid(const CgOptions<NT>& options)
{
  return options.tolerance >= NT{0} && options.max_iterations >= 0;
}

}  // namespace detail

}  // namespace spcraft
