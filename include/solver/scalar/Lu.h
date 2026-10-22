#pragma once

#include "solver/Lu.h"

namespace spcraft::solver::scalar
{

template <class IT, class NT, class OT = IT>
struct Factors {
  CscMatrix<IT, NT, OT> lower;
  CscMatrix<IT, NT, OT> upper;
  std::vector<IT> order;
  LuStatus<IT> status;
};

//! Reusable private scratch; size values and marked to n, then clear touched.
template <class IT, class NT>
struct Workspace {
  DenseVector<IT, NT> values;
  std::vector<IT> inverse_order;  //!< old -> new, constructed once by Factorize.
  std::vector<unsigned char> marked;
  std::vector<IT> touched;
};

/**
 * @brief W1.1: compute ordered column j with a left-looking sparse accumulator.
 *
 * The caller allocates L/U from symbolic and initializes workspace. Columns
 * before j are complete; only column j and workspace may be written. Scatter
 * A(order[:],order[j]), apply earlier L columns in elimination order, and store
 * U(:,j) and L(:,j). Clear every touched accumulator entry before returning.
 * Report ZeroPivot at j before dividing by a rejected pivot.
 */
template <class IT, class NT, class OT>
LuStatus<IT> FactorColumn(const CscMatrix<IT, NT, OT>& A, const LuSymbolic<IT, OT>& symbolic, IT j,
                          Factors<IT, NT, OT>& factors, Workspace<IT, NT>& workspace,
                          const LuOptions<NT>& options = {});

/**
 * @brief W1.2: validate, allocate factors, and factor columns in order.
 *
 * A is square, finite, canonical CSC. Validate dimensions, CSC bounds, triangular
 * symbolic patterns, permutation, and finite nonnegative tolerance; throw
 * invalid_argument for invalid input. Do not modify A or symbolic. The returned
 * factors own their values and metadata. Empty input succeeds. A rejected pivot
 * returns partial factors with ZeroPivot; these must not be used by Solve.
 * The assignment assumes SPD or strictly diagonally dominant input so no-pivot
 * LU is appropriate; detecting these matrix classes is not required.
 */
template <class IT, class NT, class OT>
Factors<IT, NT, OT> Factorize(const CscMatrix<IT, NT, OT>& A, const LuSymbolic<IT, OT>& symbolic,
                              const LuOptions<NT>& options = {});

//! W1.3: solve L*y=rhs in ordered coordinates; L's unit diagonal is explicit.
//! rhs and y must be distinct n-element vectors; do not change L or rhs.
template <class IT, class NT, class OT>
void ForwardSolve(const CscMatrix<IT, NT, OT>& lower, const DenseVector<IT, NT>& rhs, DenseVector<IT, NT>& y);

//! W1.4: solve U*x=rhs in ordered coordinates; reject zero/nonfinite diagonals.
//! rhs and x must be distinct n-element vectors; do not change U or rhs.
template <class IT, class NT, class OT>
void BackwardSolve(const CscMatrix<IT, NT, OT>& upper, const DenseVector<IT, NT>& rhs,
                   DenseVector<IT, NT>& x);

/**
 * @brief W1.5: solve A*x=b with permutation, forward solve, and backward solve.
 *
 * Gather rhs[i]=b[order[i]], solve L*U*z=rhs, scatter x[order[i]]=z[i]. Empty
 * order is identity. b and x are distinct preallocated n-element vectors.
 * Reject unsuccessful factors and dimension/alias errors with invalid_argument;
 * malformed/zero triangular diagonals are numerical failures (runtime_error).
 * Reuse factors for multiple calls. Empty successful factors solve empty vectors.
 */
template <class IT, class NT, class OT>
void Solve(const Factors<IT, NT, OT>& factors, const DenseVector<IT, NT>& b, DenseVector<IT, NT>& x);

// Compiled scaffold: double with signed (IT,OT)=(32,32),(32,64),(64,64).
}  // namespace spcraft::solver::scalar
