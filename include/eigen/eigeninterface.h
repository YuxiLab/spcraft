#pragma once

#include "utils/MatrixMacros.h"

namespace spcraft
{

template <class IT, class NT, class OT>
class CscMatrix;

template <class IT, class NT, class OT>
class CsrMatrix;

template <class IT, class NT>
class DenseVector;

struct EigenComparison {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  bool matches = false;
  double error = 0.0;
  // Reference multiplication time, excluding conversion and comparison.
  double multiply_seconds = 0.0;
};

template <class IT, class NT, class OT>
SPND EigenComparison CompareSpMVWithEigen(const SPCSR& a, const SPDVEC& x, const SPDVEC& y, NT tol);

template <class IT, class NT, class OT>
SPND EigenComparison CompareSpGEMMWithEigen(const SPCSC& a, const SPCSC& b, const SPCSC& c, NT tol);

}  // namespace spcraft
