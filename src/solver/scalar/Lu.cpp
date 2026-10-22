#include "solver/scalar/Lu.h"

#include <cstdint>
#include <stdexcept>

namespace spcraft::solver::scalar
{

template <class IT, class NT, class OT>
LuStatus<IT> FactorColumn(const CscMatrix<IT, NT, OT>& A, const LuSymbolic<IT, OT>& symbolic, IT j,
                          Factors<IT, NT, OT>& factors, Workspace<IT, NT>& workspace,
                          const LuOptions<NT>& options)
{
  // TODO(W1.1): see docs/solver/week1.md and the header contract.
  throw std::logic_error("W1.1 FactorColumn: implement sparse left-looking elimination");
}

template <class IT, class NT, class OT>
Factors<IT, NT, OT> Factorize(const CscMatrix<IT, NT, OT>& A, const LuSymbolic<IT, OT>& symbolic,
                              const LuOptions<NT>& options)
{
  // TODO(W1.2): see docs/solver/week1.md and the header contract.
  throw std::logic_error("W1.2 Factorize: implement validation, allocation, and the column loop");
}

template <class IT, class NT, class OT>
void ForwardSolve(const CscMatrix<IT, NT, OT>& lower, const DenseVector<IT, NT>& rhs, DenseVector<IT, NT>& y)
{
  // TODO(W1.3): see docs/solver/week1.md and the header contract.
  throw std::logic_error("W1.3 ForwardSolve: implement unit-lower CSC substitution");
}

template <class IT, class NT, class OT>
void BackwardSolve(const CscMatrix<IT, NT, OT>& upper, const DenseVector<IT, NT>& rhs, DenseVector<IT, NT>& x)
{
  // TODO(W1.4): see docs/solver/week1.md and the header contract.
  throw std::logic_error("W1.4 BackwardSolve: implement upper CSC substitution");
}

template <class IT, class NT, class OT>
void Solve(const Factors<IT, NT, OT>& factors, const DenseVector<IT, NT>& b, DenseVector<IT, NT>& x)
{
  // TODO(W1.5): see docs/solver/week1.md and the header contract.
  throw std::logic_error("W1.5 Solve: implement permutations and triangular solves");
}

// Match the assignment's compiled index/offset combinations.
template LuStatus<int32_t> FactorColumn(const CscMatrix<int32_t, double, int32_t>& A,
                                        const LuSymbolic<int32_t, int32_t>& symbolic, int32_t j,
                                        Factors<int32_t, double, int32_t>& factors,
                                        Workspace<int32_t, double>& workspace,
                                        const LuOptions<double>& options);
template Factors<int32_t, double, int32_t> Factorize(const CscMatrix<int32_t, double, int32_t>& A,
                                                     const LuSymbolic<int32_t, int32_t>& symbolic,
                                                     const LuOptions<double>& options);
template void ForwardSolve(const CscMatrix<int32_t, double, int32_t>& lower,
                           const DenseVector<int32_t, double>& rhs, DenseVector<int32_t, double>& y);
template void BackwardSolve(const CscMatrix<int32_t, double, int32_t>& upper,
                            const DenseVector<int32_t, double>& rhs, DenseVector<int32_t, double>& x);
template void Solve(const Factors<int32_t, double, int32_t>& factors, const DenseVector<int32_t, double>& b,
                    DenseVector<int32_t, double>& x);

template LuStatus<int32_t> FactorColumn(const CscMatrix<int32_t, double, int64_t>& A,
                                        const LuSymbolic<int32_t, int64_t>& symbolic, int32_t j,
                                        Factors<int32_t, double, int64_t>& factors,
                                        Workspace<int32_t, double>& workspace,
                                        const LuOptions<double>& options);
template Factors<int32_t, double, int64_t> Factorize(const CscMatrix<int32_t, double, int64_t>& A,
                                                     const LuSymbolic<int32_t, int64_t>& symbolic,
                                                     const LuOptions<double>& options);
template void ForwardSolve(const CscMatrix<int32_t, double, int64_t>& lower,
                           const DenseVector<int32_t, double>& rhs, DenseVector<int32_t, double>& y);
template void BackwardSolve(const CscMatrix<int32_t, double, int64_t>& upper,
                            const DenseVector<int32_t, double>& rhs, DenseVector<int32_t, double>& x);
template void Solve(const Factors<int32_t, double, int64_t>& factors, const DenseVector<int32_t, double>& b,
                    DenseVector<int32_t, double>& x);

template LuStatus<int64_t> FactorColumn(const CscMatrix<int64_t, double, int64_t>& A,
                                        const LuSymbolic<int64_t, int64_t>& symbolic, int64_t j,
                                        Factors<int64_t, double, int64_t>& factors,
                                        Workspace<int64_t, double>& workspace,
                                        const LuOptions<double>& options);
template Factors<int64_t, double, int64_t> Factorize(const CscMatrix<int64_t, double, int64_t>& A,
                                                     const LuSymbolic<int64_t, int64_t>& symbolic,
                                                     const LuOptions<double>& options);
template void ForwardSolve(const CscMatrix<int64_t, double, int64_t>& lower,
                           const DenseVector<int64_t, double>& rhs, DenseVector<int64_t, double>& y);
template void BackwardSolve(const CscMatrix<int64_t, double, int64_t>& upper,
                            const DenseVector<int64_t, double>& rhs, DenseVector<int64_t, double>& x);
template void Solve(const Factors<int64_t, double, int64_t>& factors, const DenseVector<int64_t, double>& b,
                    DenseVector<int64_t, double>& x);

}  // namespace spcraft::solver::scalar
