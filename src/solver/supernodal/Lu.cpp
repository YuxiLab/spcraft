#include "solver/supernodal/Lu.h"

#include <cstdint>
#include <stdexcept>

namespace spcraft::solver::supernodal
{

template <class IT, class NT, class OT>
void AssemblePanel(const CscMatrix<IT, NT, OT>& A, const SupernodalSymbolic<IT, OT>& symbolic, IT s,
                   Workspace<IT, NT, OT>& workspace)
{
  // TODO(W2.1): see docs/solver/week2.md and the header contract.
  throw std::logic_error("W2.1 AssemblePanel: implement ordered dense cross assembly");
}

template <class IT, class NT, class OT>
void UpdatePanel(const SupernodalSymbolic<IT, OT>& symbolic, IT s, const Factors<IT, NT, OT>& factors,
                 Workspace<IT, NT, OT>& workspace)
{
  // TODO(W2.2): see docs/solver/week2.md and the header contract.
  throw std::logic_error("W2.2 UpdatePanel: implement contributor block updates");
}

template <class IT, class NT, class OT>
LuStatus<IT> FactorPanel(Workspace<IT, NT, OT>& workspace, Panel<IT, NT, OT>& output,
                         const LuOptions<NT>& options)
{
  // TODO(W2.3): see docs/solver/week2.md and the header contract.
  throw std::logic_error("W2.3 FactorPanel: implement diagonal LU and block triangular solves");
}

template <class IT, class NT, class OT>
Factors<IT, NT, OT> Factorize(const CscMatrix<IT, NT, OT>& A, const SupernodalSymbolic<IT, OT>& symbolic,
                              const LuOptions<NT>& options)
{
  // TODO(W2.4): see docs/solver/week2.md and the header contract.
  throw std::logic_error("W2.4 Factorize: implement serial supernodal traversal");
}

template <class IT, class NT, class OT>
void Solve(const Factors<IT, NT, OT>& factors, const DenseVector<IT, NT>& b, DenseVector<IT, NT>& x)
{
  // TODO(W2.5): see docs/solver/week2.md and the header contract.
  throw std::logic_error("W2.5 Solve: implement block substitution and permutations");
}

// Match the assignment's compiled index/offset combinations.
template void AssemblePanel(const CscMatrix<int32_t, double, int32_t>& A,
                            const SupernodalSymbolic<int32_t, int32_t>& symbolic, int32_t s,
                            Workspace<int32_t, double, int32_t>& workspace);
template void UpdatePanel(const SupernodalSymbolic<int32_t, int32_t>& symbolic, int32_t s,
                          const Factors<int32_t, double, int32_t>& factors,
                          Workspace<int32_t, double, int32_t>& workspace);
template LuStatus<int32_t> FactorPanel(Workspace<int32_t, double, int32_t>& workspace,
                                       Panel<int32_t, double, int32_t>& output,
                                       const LuOptions<double>& options);
template Factors<int32_t, double, int32_t> Factorize(const CscMatrix<int32_t, double, int32_t>& A,
                                                     const SupernodalSymbolic<int32_t, int32_t>& symbolic,
                                                     const LuOptions<double>& options);
template void Solve(const Factors<int32_t, double, int32_t>& factors, const DenseVector<int32_t, double>& b,
                    DenseVector<int32_t, double>& x);

template void AssemblePanel(const CscMatrix<int32_t, double, int64_t>& A,
                            const SupernodalSymbolic<int32_t, int64_t>& symbolic, int32_t s,
                            Workspace<int32_t, double, int64_t>& workspace);
template void UpdatePanel(const SupernodalSymbolic<int32_t, int64_t>& symbolic, int32_t s,
                          const Factors<int32_t, double, int64_t>& factors,
                          Workspace<int32_t, double, int64_t>& workspace);
template LuStatus<int32_t> FactorPanel(Workspace<int32_t, double, int64_t>& workspace,
                                       Panel<int32_t, double, int64_t>& output,
                                       const LuOptions<double>& options);
template Factors<int32_t, double, int64_t> Factorize(const CscMatrix<int32_t, double, int64_t>& A,
                                                     const SupernodalSymbolic<int32_t, int64_t>& symbolic,
                                                     const LuOptions<double>& options);
template void Solve(const Factors<int32_t, double, int64_t>& factors, const DenseVector<int32_t, double>& b,
                    DenseVector<int32_t, double>& x);

template void AssemblePanel(const CscMatrix<int64_t, double, int64_t>& A,
                            const SupernodalSymbolic<int64_t, int64_t>& symbolic, int64_t s,
                            Workspace<int64_t, double, int64_t>& workspace);
template void UpdatePanel(const SupernodalSymbolic<int64_t, int64_t>& symbolic, int64_t s,
                          const Factors<int64_t, double, int64_t>& factors,
                          Workspace<int64_t, double, int64_t>& workspace);
template LuStatus<int64_t> FactorPanel(Workspace<int64_t, double, int64_t>& workspace,
                                       Panel<int64_t, double, int64_t>& output,
                                       const LuOptions<double>& options);
template Factors<int64_t, double, int64_t> Factorize(const CscMatrix<int64_t, double, int64_t>& A,
                                                     const SupernodalSymbolic<int64_t, int64_t>& symbolic,
                                                     const LuOptions<double>& options);
template void Solve(const Factors<int64_t, double, int64_t>& factors, const DenseVector<int64_t, double>& b,
                    DenseVector<int64_t, double>& x);

}  // namespace spcraft::solver::supernodal
