#pragma once

#include "core/CooMatrix.h"

namespace spcraft
{

template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] CooMatrix<IT, NT, OT> OmpHashSpGEMM(const CsrMatrix<IT, NT, OT>& A,
                                                  const CsrMatrix<IT, NT, OT>& B)
{
  // clang-format off
  if (A.n != B.m) throw std::invalid_argument("SpGEMM matrix dimensions do not match");
  if (A.m < 0 || A.n < 0 || B.m < 0 || B.n < 0) throw std::invalid_argument("SpGEMM dimensions must be nonnegative");
  // clang-format on
  CooMatrix<IT, NT, OT> result;
  if (!A.nnz || !B.nnz) {
    result.Allocate(0, A.m, B.n);
    return result;
  }
  fmt::print("hello we are inside the functions!");
  return result;
}

}  // namespace spcraft
