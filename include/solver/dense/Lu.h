#pragma once

#include "solver/Lu.h"

namespace spcraft::solver::dense
{

template <class IT, class NT, class OT = IT>
struct Factors {
  DenseMatrix<IT, NT, OT> lu;  // Strict-lower L and upper U; L's diagonal is implicit.
  LuStatus<IT> status;
};

template <class IT, class NT, class OT>
void ForwardSolve(const DenseMatrix<IT, NT, OT>& lu, const DenseVector<IT, NT>& rhs, DenseVector<IT, NT>& y);

template <class IT, class NT, class OT>
void BackwardSolve(const DenseMatrix<IT, NT, OT>& lu, const DenseVector<IT, NT>& rhs, DenseVector<IT, NT>& x);

template <class IT, class NT, class OT>
Factors<IT, NT, OT> RightLookingFactorize(const DenseMatrix<IT, NT, OT>& A,
                                          const LuOptions<NT>& options = {});

template <class IT, class NT, class OT>
Factors<IT, NT, OT> LeftLookingFactorize(const DenseMatrix<IT, NT, OT>& A, const LuOptions<NT>& options = {});

template <class IT, class NT, class OT>
Factors<IT, NT, OT> UpperLookingFactorize(const DenseMatrix<IT, NT, OT>& A,
                                          const LuOptions<NT>& options = {});

template <class IT, class NT, class OT>
void Solve(const Factors<IT, NT, OT>& factors, const DenseVector<IT, NT>& b, DenseVector<IT, NT>& x);

}  // namespace spcraft::solver::dense
