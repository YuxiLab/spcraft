#pragma once

#include <cstdint>

namespace spcraft
{

template <class IT, class NT, class OT>
class CscMatrix;

template <class IT, class NT, class OT>
class CsrMatrix;

template <class IT, class NT>
class DenseVector;

struct EigenComparison {
  bool matches;
  double error;
  // SpGEMM reference multiplication time, excluding the correctness comparison.
  double multiply_seconds = 0.0;
};

using EigenInterfaceIndex = std::int64_t;
using EigenInterfaceValue = double;
using EigenInterfaceCsc = CscMatrix<EigenInterfaceIndex, EigenInterfaceValue, EigenInterfaceIndex>;
using EigenInterfaceCsr = CsrMatrix<EigenInterfaceIndex, EigenInterfaceValue, EigenInterfaceIndex>;
using EigenInterfaceVector = DenseVector<EigenInterfaceIndex, EigenInterfaceValue>;

[[nodiscard]] EigenComparison CompareSpMVWithEigen(const EigenInterfaceCsr& matrix,
                                                   const EigenInterfaceVector& input,
                                                   const EigenInterfaceVector& result,
                                                   EigenInterfaceValue tolerance);

[[nodiscard]] EigenComparison CompareSpGEMMWithEigen(const EigenInterfaceCsc& lhs,
                                                     const EigenInterfaceCsc& rhs,
                                                     const EigenInterfaceCsc& result,
                                                     EigenInterfaceValue tolerance);

}  // namespace spcraft
