#include "solver/mt/Lu.h"

#include <cstdint>
#include <stdexcept>

namespace spcraft::solver::mt
{

template <class IT, class OT>
TaskGraph<IT, OT> BuildTaskGraph(const SupernodalSymbolic<IT, OT>& symbolic)
{
  // TODO(W3.1): see docs/solver/week3.md and the header contract.
  throw std::logic_error("W3.1 BuildTaskGraph: invert and validate contributor lists");
}

template <class IT, class NT, class OT>
supernodal::Factors<IT, NT, OT> Factorize(const CscMatrix<IT, NT, OT>& A,
                                          const SupernodalSymbolic<IT, OT>& symbolic,
                                          const Options<NT>& options)
{
  // TODO(W3.2): see docs/solver/week3.md and the header contract.
  throw std::logic_error("W3.2 Factorize: implement dependency-safe OpenMP scheduling");
}

// Match the assignment's compiled index/offset combinations.
template TaskGraph<int32_t, int32_t> BuildTaskGraph(const SupernodalSymbolic<int32_t, int32_t>& symbolic);
template supernodal::Factors<int32_t, double, int32_t> Factorize(
    const CscMatrix<int32_t, double, int32_t>& A, const SupernodalSymbolic<int32_t, int32_t>& symbolic,
    const Options<double>& options);

template TaskGraph<int32_t, int64_t> BuildTaskGraph(const SupernodalSymbolic<int32_t, int64_t>& symbolic);
template supernodal::Factors<int32_t, double, int64_t> Factorize(
    const CscMatrix<int32_t, double, int64_t>& A, const SupernodalSymbolic<int32_t, int64_t>& symbolic,
    const Options<double>& options);

template TaskGraph<int64_t, int64_t> BuildTaskGraph(const SupernodalSymbolic<int64_t, int64_t>& symbolic);
template supernodal::Factors<int64_t, double, int64_t> Factorize(
    const CscMatrix<int64_t, double, int64_t>& A, const SupernodalSymbolic<int64_t, int64_t>& symbolic,
    const Options<double>& options);

}  // namespace spcraft::solver::mt
