#pragma once

#include "solver/supernodal/Lu.h"

namespace spcraft::solver::mt
{

template <class NT>
struct Options {
  int threads = 1;  //!< Positive requested number of factorization workers.
  LuOptions<NT> numeric;
};

//! Successors in CSR by panel ID; indegree[s] equals contributors[s].size().
template <class IT, class OT = IT>
struct TaskGraph {
  std::vector<OT> successor_ptr;
  std::vector<IT> successor_id;
  std::vector<IT> indegree;
};

//! W3.1: invert contributor lists, rejecting duplicate, self, or non-earlier dependencies.
//! Keep symbolic unchanged. An empty graph has successor_ptr={0} and no vertices.
template <class IT, class OT>
TaskGraph<IT, OT> BuildTaskGraph(const SupernodalSymbolic<IT, OT>& symbolic);

/**
 * @brief W3.2: factor ready panels concurrently using Week 2 numerical kernels.
 *
 * Allocate the output panel vector before starting workers; never resize it
 * concurrently. Each panel has one writer and its own Workspace. Consume factors
 * only after every contributor is successfully published. Use OpenMP task
 * dependencies or a synchronized ready queue, through the project's omp wrapper.
 * Do not use a parallel-for over panel IDs. Keep BLAS single-threaded.
 *
 * On zero pivot, stop releasing dependent work, join all workers, and return
 * unusable factors. Concurrent failures report the smallest failed ordered
 * pivot column. Catch worker exceptions and rethrow after joining; exceptions
 * must not escape an OpenMP region. Use serial execution when OpenMP is disabled.
 * The result is the SAME factor representation as Week 2; reuse its serial Solve.
 * Validation and numeric failure semantics match supernodal::Factorize.
 */
template <class IT, class NT, class OT>
supernodal::Factors<IT, NT, OT> Factorize(const CscMatrix<IT, NT, OT>& A,
                                          const SupernodalSymbolic<IT, OT>& symbolic,
                                          const Options<NT>& options = {});

// Compiled scaffold: double with signed (IT,OT)=(32,32),(32,64),(64,64).
}  // namespace spcraft::solver::mt
