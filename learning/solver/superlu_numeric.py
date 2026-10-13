"""Numerical stages of the educational SuperLU pipeline.

Source guide:
  superlu/SRC/dgsequ.c, dlaqgs.c -> equilibration
  superlu/SRC/dgstrf.c           -> panels, factorization
  superlu/SRC/dpivotL.c          -> threshold partial pivoting
  superlu/SRC/dpanel_bmod.c      -> block-update idea
  superlu/SRC/dgstrs.c           -> triangular solves

The factorization here is a blocked RIGHT-looking algorithm; SuperLU uses
LEFT-looking supernodal updates. The algebraic stages are the same, but this
is not a line-by-line port of its kernels. Sparse dictionary rows carry dynamic
fill. Each panel factors its columns, solves L11*U12=A12, then performs the
real block update A22 -= L21*U12. Actual L supernodes are packed afterwards
and used by the forward/backward solves, including transpose solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
import sys
from typing import Sequence

if __package__:
    from .sparse_solver import CSR
    from .superlu_symbolic import SymbolicAnalysis, strict_supernodes
else:
    from sparse_solver import CSR
    from superlu_symbolic import SymbolicAnalysis, strict_supernodes


@dataclass(frozen=True)
class Scaling:
    row: tuple[float, ...]
    column: tuple[float, ...]
    equed: str  # N=none, R=rows, C=columns, B=both.


def equilibrate(A: CSR, mode: str = "auto") -> tuple[CSR, Scaling]:
    """Compute max-norm row/column scales, then apply the selected scales.

    'auto' follows dlaqgs's 0.1 ratio / safe-magnitude decision. 'both' always
    applies both candidate scales; 'none' preserves the original values.
    """
    if mode not in ("auto", "both", "none"):
        raise ValueError("equilibration must be 'auto', 'both', or 'none'")
    n = A.shape[0]
    ones = (1.0,) * n
    if mode == "none" or n == 0:
        return A, Scaling(ones, ones, "N")

    tiny, huge = sys.float_info.min, 1.0 / sys.float_info.min
    row_max = [max((abs(v) for v in A.data[A.indptr[i]:A.indptr[i + 1]]),
                   default=0.0) for i in range(n)]
    if any(value == 0.0 for value in row_max):
        raise ValueError("Singular matrix: zero row during equilibration")
    row_scale = [1.0 / min(max(value, tiny), huge) for value in row_max]
    col_max = [0.0] * n
    for i in range(n):
        for k in range(A.indptr[i], A.indptr[i + 1]):
            j = A.indices[k]
            col_max[j] = max(col_max[j], abs(A.data[k]) * row_scale[i])
    if any(value == 0.0 for value in col_max):
        raise ValueError("Singular matrix: zero column during equilibration")
    col_scale = [1.0 / min(max(value, tiny), huge) for value in col_max]
    row_ratio = max(min(row_max), tiny) / min(max(row_max), huge)
    col_ratio = max(min(col_max), tiny) / min(max(col_max), huge)
    small = tiny / sys.float_info.epsilon
    amax = max(row_max)
    use_rows = mode == "both" or row_ratio < 0.1 or not small <= amax <= 1.0 / small
    use_columns = mode == "both" or col_ratio < 0.1
    row = tuple(row_scale) if use_rows else ones
    col = tuple(col_scale) if use_columns else ones
    scaled = tuple(
        (A.data[k] * row[i]) * col[A.indices[k]]
        for i in range(n)
        for k in range(A.indptr[i], A.indptr[i + 1])
    )
    if any(not isfinite(v) for v in scaled):
        raise ArithmeticError("Equilibration produced nonfinite entries")
    if any(v == 0 for v in scaled):
        raise ArithmeticError("Equilibration underflowed a nonzero entry")
    equed = "B" if use_rows and use_columns else "R" if use_rows else "C" if use_columns else "N"
    return CSR(A.shape, scaled, A.indices, A.indptr), Scaling(row, col, equed)


@dataclass
class NumericStats:
    panels: int = 0
    row_swaps: int = 0
    reused_pivots: int = 0
    diagonal_pivots: int = 0
    fill_insertions: int = 0  # Cumulative creations, not final structural fill.
    multiply_adds: int = 0
    peak_work_nnz: int = 0
    current_work_nnz: int = 0


def _put(row: dict[int, float], j: int, value: float, stats: NumericStats) -> None:
    if not isfinite(value):
        raise ArithmeticError("Numerical overflow during factorization")
    if value == 0.0:
        if j in row:
            del row[j]
            stats.current_work_nnz -= 1
    else:
        if j not in row:
            stats.fill_insertions += 1
            stats.current_work_nnz += 1
            stats.peak_work_nnz = max(stats.peak_work_nnz, stats.current_work_nnz)
        row[j] = value


def factor_panels(
    A: CSR,
    plan: SymbolicAnalysis,
    *,
    pivot_threshold: float = 1.0,
    pivot_atol: float = 0.0,
    preferred_row_order: Sequence[int] | None = None,
    trace: bool = False,
) -> tuple[list[dict[int, float]], list[dict[int, float]], list[int], NumericStats]:
    """Factor P_r*(scaled A)[:,col_order]=L*U, with delayed block updates."""
    n = A.shape[0]
    if not isfinite(pivot_threshold) or not 0 <= pivot_threshold <= 1:
        raise ValueError("pivot_threshold must lie in [0,1]")
    if not isfinite(pivot_atol) or pivot_atol < 0:
        raise ValueError("pivot_atol must be finite and nonnegative")
    if preferred_row_order is not None and sorted(preferred_row_order) != list(range(n)):
        raise ValueError("preferred_row_order must be a row permutation")

    inverse_columns = [0] * n
    for new, old in enumerate(plan.col_order):
        inverse_columns[old] = new
    work = [
        {inverse_columns[A.indices[k]]: A.data[k]
         for k in range(A.indptr[i], A.indptr[i + 1])}
        for i in range(n)
    ]
    lower: list[dict[int, float]] = [{} for _ in range(n)]
    rows = list(range(n))
    # Track original-row locations so diagonal and reused pivots are O(1) lookups.
    row_position = list(range(n))
    stats = NumericStats(peak_work_nnz=len(A.data), current_work_nnz=len(A.data))

    for start, end in plan.panels:
        stats.panels += 1
        if trace:
            print(f"  factor panel [{start}, {end})")
        for k in range(start, end):
            largest_row = max(range(k, n), key=lambda i: abs(work[i].get(k, 0.0)))
            largest = abs(work[largest_row].get(k, 0.0))
            if largest <= pivot_atol:
                raise ValueError(f"Singular or rejected pivot at ordered column {k}")
            threshold = pivot_threshold * largest
            pivot_row = largest_row
            reused = False
            if preferred_row_order is not None:
                candidate = row_position[preferred_row_order[k]]
                value = abs(work[candidate].get(k, 0.0))
                if candidate >= k and value > pivot_atol and value >= threshold:
                    pivot_row, reused = candidate, True
                    stats.reused_pivots += 1
            if not reused:
                # The original diagonal belongs to row col_order[k], not row k.
                candidate = row_position[plan.col_order[k]]
                value = abs(work[candidate].get(k, 0.0))
                if candidate >= k and value > pivot_atol and value >= threshold:
                    pivot_row = candidate
                    stats.diagonal_pivots += 1
            if pivot_row != k:
                work[k], work[pivot_row] = work[pivot_row], work[k]
                lower[k], lower[pivot_row] = lower[pivot_row], lower[k]
                rows[k], rows[pivot_row] = rows[pivot_row], rows[k]
                row_position[rows[k]], row_position[rows[pivot_row]] = k, pivot_row
                stats.row_swaps += 1
            pivot = work[k][k]
            if trace:
                print(f"    col={k}, original pivot row={rows[k]}, value={pivot:.6g}")
            within_panel = [(j, value) for j, value in work[k].items() if k < j < end]
            for i in range(k + 1, n):
                value = work[i].pop(k, 0.0)
                if value == 0.0:
                    continue
                stats.current_work_nnz -= 1
                multiplier = value / pivot
                if not isfinite(multiplier):
                    raise ArithmeticError("Pivot division overflow")
                lower[i][k] = multiplier
                for j, upper in within_panel:
                    _put(work[i], j, work[i].get(j, 0.0) - multiplier * upper, stats)
                    stats.multiply_adds += 1

        # Block triangular solve: L11*U12=A12. Trailing columns have deliberately
        # not been updated by scalar elimination inside the panel.
        for i in range(start, end):
            coefficients = [(k, lower[i][k]) for k in range(start, i) if k in lower[i]]
            touched = {j for k, _ in coefficients for j in work[k] if j >= end}
            for j in touched:
                contribution = fsum(lik * work[k].get(j, 0.0) for k, lik in coefficients)
                _put(work[i], j, work[i].get(j, 0.0) - contribution, stats)
                stats.multiply_adds += len(coefficients)

        # Sparse block Schur update: A22 -= L21*U12. The inner dimension is
        # the panel width; cancellations and dynamic fill are handled explicitly.
        for i in range(end, n):
            coefficients = [(k, lower[i][k]) for k in range(start, end) if k in lower[i]]
            touched = {j for k, _ in coefficients for j in work[k] if j >= end}
            for j in touched:
                contribution = fsum(lik * work[k].get(j, 0.0) for k, lik in coefficients)
                _put(work[i], j, work[i].get(j, 0.0) - contribution, stats)
                stats.multiply_adds += len(coefficients)
    return lower, work, rows, stats


@dataclass(frozen=True)
class Supernode:
    start: int
    end: int
    # Dense diagonal blocks; lower unit diagonal is explicit in this packed form.
    L11: tuple[tuple[float, ...], ...]
    U11: tuple[tuple[float, ...], ...]
    below_rows: tuple[int, ...]
    L21: tuple[tuple[float, ...], ...]
    trailing_columns: tuple[int, ...]
    U12: tuple[tuple[float, ...], ...]

    @property
    def stored_values(self) -> int:
        width = self.end - self.start
        return 2 * width * width + width * (len(self.below_rows) + len(self.trailing_columns))


def pack_supernodes(
    lower: Sequence[dict[int, float]], upper: Sequence[dict[int, float]], max_width: int
) -> tuple[Supernode, ...]:
    """Detect actual numeric L column structure and pack genuine dense blocks.

    Pivoting may invalidate the symbolic supernode guesses, so detection uses
    the completed factors. These blocks are subsequently used for solves.
    """
    n = len(lower)
    column_rows = [{j} for j in range(n)]
    for i, row in enumerate(lower):
        for j, value in row.items():
            if value != 0.0:
                column_rows[j].add(i)
    nodes = []
    for start, end in strict_supernodes(column_rows, max_width):
        below = sorted(set().union(*column_rows[start:end]) - set(range(start, end)))
        trailing = sorted({j for i in range(start, end) for j in upper[i] if j >= end})
        L11 = tuple(tuple(1.0 if i == j else lower[i].get(j, 0.0)
                          for j in range(start, end)) for i in range(start, end))
        U11 = tuple(tuple(upper[i].get(j, 0.0) for j in range(start, end))
                    for i in range(start, end))
        L21 = tuple(tuple(lower[i].get(j, 0.0) for j in range(start, end)) for i in below)
        U12 = tuple(tuple(upper[i].get(j, 0.0) for j in trailing) for i in range(start, end))
        nodes.append(Supernode(start, end, L11, U11, tuple(below), L21,
                               tuple(trailing), U12))
    return tuple(nodes)


def triangular_solve(nodes: Sequence[Supernode], rhs: Sequence[float], *, transpose: bool = False) -> list[float]:
    """Solve L*U*z=rhs or U^T*L^T*z=rhs using the packed supernodes."""
    n = len(rhs)
    work = list(rhs)
    intermediate = [0.0] * n
    if not transpose:
        # Forward: solve a dense L11 block, then apply its L21 update.
        for node in nodes:
            s, e = node.start, node.end
            width = e - s
            for a in range(width):
                intermediate[s + a] = work[s + a] - fsum(
                    node.L11[a][b] * intermediate[s + b] for b in range(a)
                )
            for row, coefficients in zip(node.below_rows, node.L21):
                work[row] -= fsum(coefficients[b] * intermediate[s + b] for b in range(width))
        # Backward: subtract U12 times known variables, then solve U11.
        result = [0.0] * n
        for node in reversed(nodes):
            s, e = node.start, node.end
            width = e - s
            local = [intermediate[s + a] - fsum(
                value * result[j] for j, value in zip(node.trailing_columns, node.U12[a])
            ) for a in range(width)]
            for a in range(width - 1, -1, -1):
                result[s + a] = (local[a] - fsum(
                    node.U11[a][b] * result[s + b] for b in range(a + 1, width)
                )) / node.U11[a][a]
    else:
        # Forward solve U^T, including scatter updates from U12^T.
        for node in nodes:
            s, e = node.start, node.end
            width = e - s
            for a in range(width):
                intermediate[s + a] = (work[s + a] - fsum(
                    node.U11[b][a] * intermediate[s + b] for b in range(a)
                )) / node.U11[a][a]
            for b, j in enumerate(node.trailing_columns):
                work[j] -= fsum(node.U12[a][b] * intermediate[s + a] for a in range(width))
        # Backward solve L^T, with L21^T contributions and unit diagonal.
        result = [0.0] * n
        for node in reversed(nodes):
            s, e = node.start, node.end
            width = e - s
            local = [intermediate[s + a] - fsum(
                coefficients[a] * result[i] for i, coefficients in zip(node.below_rows, node.L21)
            ) for a in range(width)]
            for a in range(width - 1, -1, -1):
                result[s + a] = local[a] - fsum(
                    node.L11[b][a] * result[s + b] for b in range(a + 1, width)
                )
    if not all(isfinite(value) for value in result):
        raise ArithmeticError("Nonfinite triangular-solve result")
    return result
