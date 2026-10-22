#pragma once

#include "solver/Lu.h"

namespace spcraft::solver::supernodal
{

/**
 * @brief One numerical supernode, using existing owning dense-vector buffers.
 *
 * Let w=last-first, r=rows.size(), c=columns.size(). All blocks are column-major:
 * diagonal has w*w entries, lower has r*w entries, upper has w*c entries.
 * Before factorization they hold the updated A11/A21/A12 cross. Afterwards,
 * diagonal packs strict-lower L11 (unit diagonal implicit) and upper U11;
 * lower holds L21 and upper holds U12. Dense buffer lengths use OT.
 */
template <class IT, class NT, class OT = IT>
struct Panel {
  IT first = 0;
  IT last = 0;
  std::vector<IT> rows;
  std::vector<IT> columns;
  DenseVector<OT, NT> diagonal;
  DenseVector<OT, NT> lower;
  DenseVector<OT, NT> upper;
};

template <class IT, class NT, class OT = IT>
struct Factors {
  IT n = 0;
  std::vector<IT> order;
  std::vector<Panel<IT, NT, OT>> panels;
  LuStatus<IT> status;
};

//! A panel's private numerical workspace; never share it between active tasks.
template <class IT, class NT, class OT = IT>
struct Workspace {
  Panel<IT, NT, OT> panel;
  DenseVector<OT, NT> product;
};

//! W2.1: allocate/zero the cross for panel s and gather entries of P^T*A*P.
//! Preserve structural zeros; overflow-check block sizes before allocating.
template <class IT, class NT, class OT>
void AssemblePanel(const CscMatrix<IT, NT, OT>& A, const SupernodalSymbolic<IT, OT>& symbolic, IT s,
                   Workspace<IT, NT, OT>& workspace);

/**
 * @brief W2.2: apply completed contributors to this workspace's cross.
 *
 * For each listed contributor k, gather matching factor rows/columns and apply
 * A(R,C) -= L(R,K)*U(K,C) to A11, A21, and A12 as needed. Read ONLY the listed,
 * completed source panels; do not update unrelated trailing-matrix storage.
 * The caller guarantees that contributors are complete. Mapping and temporary
 * products belong to workspace. Use actual block updates during factorization.
 */
template <class IT, class NT, class OT>
void UpdatePanel(const SupernodalSymbolic<IT, OT>& symbolic, IT s, const Factors<IT, NT, OT>& factors,
                 Workspace<IT, NT, OT>& workspace);

/**
 * @brief W2.3: factor A11, solve L21*U11=A21 and L11*U12=A12, publish output.
 *
 * No pivoting. On a rejected diagonal return ZeroPivot with the GLOBAL ordered
 * column. On success move the finished buffers into output. This function writes
 * only workspace and output, and must not publish global completion itself.
 * Dense scalar kernels are acceptable initially; explain TRSM/GEMM equivalents.
 */
template <class IT, class NT, class OT>
LuStatus<IT> FactorPanel(Workspace<IT, NT, OT>& workspace, Panel<IT, NT, OT>& output,
                         const LuOptions<NT>& options = {});

//! W2.4: validate the supplied plan and factor panels serially in ID order.
//! Same input/error contract as scalar::Factorize; no scalar-factor-then-pack shortcut.
template <class IT, class NT, class OT>
Factors<IT, NT, OT> Factorize(const CscMatrix<IT, NT, OT>& A, const SupernodalSymbolic<IT, OT>& symbolic,
                              const LuOptions<NT>& options = {});

//! W2.5: serial block forward/backward solves, including original-coordinate permutations.
//! Same dimensions, ownership, reuse, and failure contract as scalar::Solve.
template <class IT, class NT, class OT>
void Solve(const Factors<IT, NT, OT>& factors, const DenseVector<IT, NT>& b, DenseVector<IT, NT>& x);

// Compiled scaffold: double with signed (IT,OT)=(32,32),(32,64),(64,64).
}  // namespace spcraft::solver::supernodal
