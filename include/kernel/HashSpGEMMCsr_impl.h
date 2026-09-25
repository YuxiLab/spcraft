#pragma once

#include "kernel/mtSpGEMM.h"

namespace spcraft
{

SP_SR_MAT_TEMP
SPND SPCOO OmpHashSpGEMM(const SPCSR& A, const SPCSR& B)
{
  if (A.n != B.m) throw std::invalid_argument("SpGEMM matrix dimensions do not match");
  if (A.m < 0 || A.n < 0 || B.m < 0 || B.n < 0)
    throw std::invalid_argument("SpGEMM dimensions must be nonnegative");
  SPCOO result;
  if (!A.nnz || !B.nnz) {
    result.Allocate(0, A.m, B.n);
    return result;
  }
  return result;
}

}  // namespace spcraft
