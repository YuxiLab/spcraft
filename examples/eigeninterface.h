#pragma once

#include "utils/MatrixMacros.h"

namespace spcraft
{

SP_MAT_TEMP
class CscMatrix;

SP_MAT_TEMP
class CsrMatrix;

SP_VEC_TEMP
class DenseVector;

struct EigenComparison {
  /*****************************************************************************
   *                                Data Members                               *
   ****************************************************************************/
  bool matches = false;
  double error = 0.0;
  // Reference multiplication time, excluding conversion and comparison.
  double multiply_seconds = 0.0;
};

SP_MAT_TEMP
SPND EigenComparison CompareSpMVWithEigen(const SPCSR& a, const SPDVEC& x, const SPDVEC& y, NT tol);

SP_MAT_TEMP
SPND EigenComparison CompareSpGEMMWithEigen(const SPCSC& a, const SPCSC& b, const SPCSC& c, NT tol);

}  // namespace spcraft
