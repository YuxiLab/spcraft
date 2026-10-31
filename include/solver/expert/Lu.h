#pragma once

#include "solver/pivoted/Lu.h"

namespace spcraft::solver::expert
{

template <class NT>
struct Scaling {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  std::vector<NT> rows;
  std::vector<NT> columns;
};

enum class RefinementCode { NotRun, Converged, Stagnated, IterationLimit, Nonfinite };

template <class NT>
struct Diagnostics {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  NT reciprocal_condition = NT{0};
  NT reciprocal_pivot_growth = NT{0};
  std::vector<NT> backward_error;       //!< Componentwise error per RHS in the ORIGINAL system.
  std::vector<NT> forward_error_bound;  //!< Estimated relative infinity-norm error per RHS.
  std::vector<RefinementCode> refinement;
  std::vector<int32_t> refinement_steps;
};

template <class NT>
struct RefinementOptions {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  int32_t max_iterations = 5;
  NT backward_tolerance = NT{0};  //!< Zero selects a machine-precision-based stopping target.
};

// Holds a reusable analysis and factors of diag(rows)*A*diag(columns).
template <class IT, class NT, class OT = IT>
struct State {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  symbolic::Analysis<IT, OT> analysis;
  Scaling<NT> scaling;
  pivoted::Factors<IT, NT, OT> factors;
};

template <class IT, class NT, class OT>
Scaling<NT> ComputeScaling(const CscMatrix<IT, NT, OT>& A);

template <class IT, class NT, class OT>
CscMatrix<IT, NT, OT> ApplyScaling(const CscMatrix<IT, NT, OT>& A, const Scaling<NT>& scaling);

// Replaces numerical state using the saved analysis. Reject a changed CSC structure.
// Old pivot rows may be proposed through options.row_hint; they are never forced past the threshold.
template <class IT, class NT, class OT>
void Refactorize(const CscMatrix<IT, NT, OT>& A, State<IT, NT, OT>& state,
                 const pivoted::Options<IT, NT>& options = {});

// Scaling-aware solve; Refine separately receives the original unscaled A and B.
template <class IT, class NT, class OT>
void Solve(const State<IT, NT, OT>& state, const DenseMatrix<IT, NT, OT>& B, DenseMatrix<IT, NT, OT>& X,
           pivoted::Operation operation = pivoted::Operation::Normal);

// Returns a reciprocal 1-norm condition estimate for the ORIGINAL A, not the scaled factors.
template <class IT, class NT, class OT>
NT EstimateReciprocalCondition(const CscMatrix<IT, NT, OT>& A, const State<IT, NT, OT>& state);

// Compare the scaled, column-permuted input with U. Return min column max(A)/max(U).
template <class IT, class NT, class OT>
NT ReciprocalPivotGrowth(const CscMatrix<IT, NT, OT>& A, const State<IT, NT, OT>& state);

template <class IT, class NT, class OT>
Diagnostics<NT> Refine(const CscMatrix<IT, NT, OT>& A, const State<IT, NT, OT>& state,
                       const DenseMatrix<IT, NT, OT>& B, DenseMatrix<IT, NT, OT>& X,
                       const RefinementOptions<NT>& options = {},
                       pivoted::Operation operation = pivoted::Operation::Normal);

}  // namespace spcraft::solver::expert
