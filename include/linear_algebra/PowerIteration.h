#pragma once

namespace spcraft
{

/**
 * @brief Stopping controls for a power iteration.
 */
template <class NT>
struct PowerIterationOptions {
  NT tolerance = NT{1e-10};   //!< Target for ||A x - lambda x|| / |lambda|.
  int max_iterations = 1000;  //!< Upper bound on iterations.
};

/**
 * @brief What a power iteration did.
 */
template <class NT>
struct PowerIterationResult {
  int iterations = 0;      //!< Iterations actually performed.
  NT eigenvalue = NT{0};   //!< Rayleigh quotient of the final iterate.
  NT residual = NT{0};     //!< Final ||A x - lambda x||, relative to |lambda|.
  bool converged = false;  //!< Whether the residual reached the tolerance.
};

namespace detail
{

//! Reject option sets no iteration can act on.
template <class NT>
[[nodiscard]] constexpr bool PowerIterationOptionsAreValid(
    const PowerIterationOptions<NT>& options)
{
  return options.tolerance >= NT{0} && options.max_iterations >= 0;
}

}  // namespace detail

}  // namespace spcraft
