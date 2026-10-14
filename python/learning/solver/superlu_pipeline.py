"""An inspectable Python implementation of the main SuperLU driver stages.

Read superlu/SRC/dgssvx.c alongside this file. Stage correspondence:

  Input / CSC            superlu_symbolic.CSC
  dgsequ + dlaqgs         superlu_numeric.equilibrate
  get_perm_c              superlu_symbolic.minimum_degree
  sp_preorder             superlu_symbolic.analyze (etree + postorder)
  relax_snode             superlu_symbolic.relaxed_supernodes
  dgstrf + dpivotL        superlu_numeric.factor_panels
  dgstrs                  superlu_numeric.triangular_solve
  dgsrfs                  SparseDirectSolver.refine, backward/forward estimates
  dgscon + dlacon2         estimate_norm_1, reciprocal_condition_estimate
  dPivotGrowth            SparseDirectSolver.reciprocal_pivot_growth
  SamePattern / FACTORED   refactor / solve_many

These are educational implementations, not an API-compatible SuperLU port.
Exact minimum degree replaces COLAMD; blocked right-looking elimination
replaces SuperLU's left-looking kernels. Dynamic sparse storage and packed
supernode solves are real. Python owns allocation; statistics report numerical
entries rather than claiming native allocator byte counts. All values are real.

Run: python3 sparse_learning/superlu_pipeline.py --grid 8 --trace
     python3 sparse_learning/superlu_pipeline.py --example pivot --trace
     python3 sparse_learning/superlu_pipeline.py --example scaled --grid 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from math import fsum, isfinite, sin
import sys
from time import perf_counter
from typing import Callable, Sequence

if __package__:
    from .sparse_solver import CSR, norm, poisson_2d
    from .sparse_lu import pivoting_example
    from .superlu_symbolic import CSC, SymbolicAnalysis, analyze, canonicalize
    from .superlu_numeric import Scaling, NumericStats, equilibrate, factor_panels, pack_supernodes, triangular_solve
else:
    from sparse_solver import CSR, norm, poisson_2d
    from sparse_lu import pivoting_example
    from superlu_symbolic import CSC, SymbolicAnalysis, analyze, canonicalize
    from superlu_numeric import Scaling, NumericStats, equilibrate, factor_panels, pack_supernodes, triangular_solve


def estimate_norm_1(
    n: int,
    apply: Callable[[Sequence[float]], list[float]],
    apply_transpose: Callable[[Sequence[float]], list[float]],
    maxiter: int = 5,
) -> float:
    """Hager/Higham-style 1-norm estimate via an operator and its transpose.

    No inverse matrix is formed. The estimate is not a certified upper bound.
    The final alternating-vector probe follows the idea used in dlacon2.
    """
    if n == 0:
        return 0.0
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    x = [1.0 / n] * n
    estimate, previous_signs, previous_index = 0.0, None, None
    for _ in range(maxiter):
        y = apply(x)
        candidate = fsum(abs(value) for value in y)
        estimate = max(estimate, candidate)
        signs = [1.0 if value >= 0 else -1.0 for value in y]
        if signs == previous_signs:
            break
        z = apply_transpose(signs)
        j = max(range(n), key=lambda i: abs(z[i]))
        if abs(z[j]) <= fsum(zi * xi for zi, xi in zip(z, x)) or j == previous_index:
            break
        x = [0.0] * n
        x[j] = 1.0
        previous_signs, previous_index = signs, j
    if n > 1:
        alternate = [(-1.0 if i % 2 else 1.0) * (1.0 + i / (n - 1)) for i in range(n)]
        estimate = max(estimate, 2.0 * fsum(abs(v) for v in apply(alternate)) / (3.0 * n))
    return estimate


@dataclass
class SolveReport:
    x: list[float]
    refinement_steps: int
    backward_error_history: list[float]
    relative_residual: float
    normwise_backward_error: float
    componentwise_backward_error: float
    forward_error_estimate: float
    reciprocal_condition_estimate: float
    reciprocal_pivot_growth: float


@dataclass
class SparseDirectSolver:
    A: CSR  # Canonical, original unscaled matrix, preserved for residuals.
    csc: CSC
    scaled_A: CSR
    scaling: Scaling
    analysis: SymbolicAnalysis
    L: list[dict[int, float]]
    U: list[dict[int, float]]
    row_order: list[int]  # New-to-old; inverse convention to SuperLU perm_r.
    supernodes: tuple
    U_csc: CSC
    numeric_stats: NumericStats
    equilibration: str
    pivot_threshold: float
    pivot_atol: float
    symbolic_reused: bool = False
    stage_seconds: dict[str, float] = field(default_factory=dict)
    solve_calls: int = 0
    _rcond: float | None = None

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def nnz_L(self) -> int:
        return self.n + sum(sum(v != 0.0 for v in row.values()) for row in self.L)

    @property
    def nnz_U(self) -> int:
        return sum(len(row) for row in self.U)

    @property
    def packed_values(self) -> int:
        return sum(node.stored_values for node in self.supernodes)

    def solve(self, b: Sequence[float], *, transpose: bool = False) -> list[float]:
        """Solve A*x=b or A^T*x=b using existing factors and scaling/permutations.

        If B=R*A*C, then P_r*B[:,q]=L*U. Normal solves scale/permutate b
        by R and P_r, then scatter the solution by q and multiply by C.
        Transpose solves reverse those roles. A is not assumed symmetric.
        """
        if len(b) != self.n:
            raise ValueError("RHS length must match the matrix")
        b = [float(value) for value in b]
        if not all(isfinite(value) for value in b):
            raise ValueError("RHS values must be finite")
        start = perf_counter()
        q, p = self.analysis.col_order, self.row_order
        if not transpose:
            rhs = [self.scaling.row[i] * b[i] for i in p]
            z = triangular_solve(self.supernodes, rhs)
            x = [0.0] * self.n
            for k, old in enumerate(q):
                x[old] = self.scaling.column[old] * z[k]
        else:
            rhs = [self.scaling.column[j] * b[j] for j in q]
            z = triangular_solve(self.supernodes, rhs, transpose=True)
            x = [0.0] * self.n
            for k, old in enumerate(p):
                x[old] = self.scaling.row[old] * z[k]
        if not all(isfinite(value) for value in x):
            raise ArithmeticError("Overflow unscaling the solution")
        self.solve_calls += 1
        self.stage_seconds["triangular_solves"] = self.stage_seconds.get("triangular_solves", 0.0) + perf_counter() - start
        return x

    def solve_many(self, rhs_columns: Sequence[Sequence[float]], *, transpose: bool = False) -> list[list[float]]:
        """Accept a sequence of RHS columns, reuse one factorization for all."""
        return [self.solve(b, transpose=transpose) for b in rhs_columns]

    def refactor(self, A: CSR, *, reuse_row_order: bool = True, trace: bool = False) -> SparseDirectSolver:
        """Same-pattern numerical refactorization; return new factors.

        The cached column ordering / etree / panel plan is reused. Scaling is
        recomputed. Old pivot rows are preferred only if they pass the current
        threshold test; this does not blindly assume the old row order is safe.
        """
        return factorize(
            A, analysis=self.analysis, equilibration=self.equilibration,
            pivot_threshold=self.pivot_threshold, pivot_atol=self.pivot_atol,
            preferred_row_order=self.row_order if reuse_row_order else None, trace=trace,
        )

    def operator_rows(self, *, transpose: bool = False) -> list[list[tuple[int, float]]]:
        if not transpose:
            return [[(self.A.indices[k], self.A.data[k])
                     for k in range(self.A.indptr[i], self.A.indptr[i + 1])]
                    for i in range(self.n)]
        return [[(self.csc.row_indices[k], self.csc.data[k])
                 for k in range(self.csc.col_ptr[j], self.csc.col_ptr[j + 1])]
                for j in range(self.n)]

    def reciprocal_condition_estimate(self) -> float:
        """Estimate 1/(||A||_1 ||A^{-1}||_1) for ORIGINAL, unscaled A."""
        if self._rcond is None:
            start = perf_counter()
            inverse_norm = estimate_norm_1(
                self.n, self.solve, lambda b: self.solve(b, transpose=True)
            )
            matrix_norm = self.csc.norm_1()
            self._rcond = (1.0 if self.n == 0 else
                           min(1.0, (1.0 / matrix_norm) / inverse_norm))
            self.stage_seconds["condition_estimate"] = perf_counter() - start
        return self._rcond

    def reciprocal_pivot_growth(self) -> float:
        """min_j max_i|scaled_A[i,q[j]]| / max_i|U[i,j]|, capped at 1."""
        maxima_A, maxima_U = [0.0] * self.n, [0.0] * self.n
        for i in range(self.n):
            for k in range(self.scaled_A.indptr[i], self.scaled_A.indptr[i + 1]):
                j = self.scaled_A.indices[k]
                maxima_A[j] = max(maxima_A[j], abs(self.scaled_A.data[k]))
        for row in self.U:
            for j, value in row.items():
                maxima_U[j] = max(maxima_U[j], abs(value))
        return min((min(1.0, maxima_A[old] / maxima_U[new])
                    for new, old in enumerate(self.analysis.col_order)), default=1.0)

    def refine(
        self,
        b: Sequence[float],
        x: Sequence[float] | None = None,
        *,
        transpose: bool = False,
        maxiter: int = 5,
        residual_precision: str = "decimal",
    ) -> SolveReport:
        """Refine a solution and estimate errors using existing LU factors.

        Like dgsrfs, stop at machine epsilon, lack of factor-two backward-error
        improvement, or the iteration limit. Optional 80-digit Decimal residuals
        illustrate higher-precision refinement; factors remain float64.
        Estimates are diagnostics, not rigorous error bounds.
        """
        if not isinstance(maxiter, int) or maxiter < 0:
            raise ValueError("maxiter must be a nonnegative integer")
        if residual_precision not in ("double", "decimal"):
            raise ValueError("residual_precision must be 'double' or 'decimal'")
        if len(b) != self.n or not all(isfinite(value) for value in b):
            raise ValueError("RHS must be a matching finite vector")
        b = [float(value) for value in b]
        x = self.solve(b, transpose=transpose) if x is None else [float(value) for value in x]
        if len(x) != self.n or not all(isfinite(value) for value in x):
            raise ValueError("Initial solution must be a matching finite vector")
        start = perf_counter()
        rows = self.operator_rows(transpose=transpose)
        history, steps, previous = [], 0, 3.0
        eps, tiny = sys.float_info.epsilon, sys.float_info.min
        safe1 = (max((len(row) for row in rows), default=0) + 1) * tiny
        safe2 = safe1 / eps
        while True:
            residual, denominators = residual_and_denominator(rows, b, x, residual_precision)
            ratios = [
                (abs(ri) / denominator if denominator > safe2 else
                 (abs(ri) + safe1) / denominator if denominator != 0 else
                 0.0 if ri == 0 else float("inf"))
                for ri, denominator in zip(residual, denominators)
            ]
            berr = max(ratios, default=0.0)
            history.append(berr)
            if berr <= eps or 2 * berr > previous or steps >= maxiter:
                break
            correction = self.solve(residual, transpose=transpose)
            x = [xi + di for xi, di in zip(x, correction)]
            previous, steps = berr, steps + 1
        self.stage_seconds["refinement"] = self.stage_seconds.get("refinement", 0.0) + perf_counter() - start

        # Forward-error estimate: || |op(A)^-1| W ||_infinity / ||x||_infinity.
        # Estimate the infinity norm as a transpose operator's 1-norm; W uses
        # the dgsrfs-style residual + rounding-error weights.
        weights = [abs(ri) + (len(row) + 1) * eps * denom + (safe1 if denom <= safe2 else 0.0)
                   for ri, row, denom in zip(residual, rows, denominators)]
        xnorm = max((abs(v) for v in x), default=0.0)
        if xnorm == 0.0 and all(value == 0.0 for value in b):
            # For nonsingular A, the homogeneous problem has exactly x=0.
            # Safe-minimum weights should not manufacture a relative error.
            ferr = 0.0
        else:
            forward = estimate_norm_1(
                self.n,
                lambda v: [wi * zi for wi, zi in zip(weights, self.solve(v, transpose=not transpose))],
                lambda v: self.solve([wi * vi for wi, vi in zip(weights, v)], transpose=transpose),
            )
            ferr = forward / xnorm if xnorm else float("inf")
        anorm_inf = max((fsum(abs(value) for _, value in row) for row in rows), default=0.0)
        bnorm_inf = max((abs(v) for v in b), default=0.0)
        normwise_denominator = anorm_inf * xnorm + bnorm_inf
        rnorm_inf = max((abs(v) for v in residual), default=0.0)
        normwise = rnorm_inf / normwise_denominator if normwise_denominator else 0.0
        bnorm = norm(b)
        relative_residual = norm(residual) / bnorm if bnorm else norm(residual)
        return SolveReport(x, steps, history, relative_residual, normwise, berr, ferr,
                           self.reciprocal_condition_estimate(), self.reciprocal_pivot_growth())


def residual_and_denominator(
    rows: Sequence[Sequence[tuple[int, float]]],
    b: Sequence[float], x: Sequence[float], precision: str,
) -> tuple[list[float], list[float]]:
    residual = []
    if precision == "decimal":
        with localcontext() as context:
            context.prec = 80
            dx = [Decimal.from_float(float(value)) for value in x]
            for row, bi in zip(rows, b):
                product = sum((Decimal.from_float(float(value)) * dx[j]
                               for j, value in row), Decimal(0))
                residual.append(float(Decimal.from_float(float(bi)) - product))
    else:
        residual = [fsum([float(bi)] + [-value * x[j] for j, value in row])
                    for row, bi in zip(rows, b)]
    denominators = [fsum([abs(bi)] + [abs(value) * abs(x[j]) for j, value in row])
                    for row, bi in zip(rows, b)]
    if not all(isfinite(v) for v in residual + denominators):
        raise ArithmeticError("Nonfinite residual or backward-error denominator")
    return residual, denominators


def factorize(
    A: CSR,
    *,
    analysis: SymbolicAnalysis | None = None,
    ordering: str = "minimum_degree",
    postorder: bool = True,
    panel_size: int = 4,
    relax: int = 4,
    equilibration: str = "auto",
    pivot_threshold: float = 1.0,
    pivot_atol: float = 0.0,
    preferred_row_order: Sequence[int] | None = None,
    trace: bool = False,
) -> SparseDirectSolver:
    """Driver for input conversion, scaling, structural analysis, and factorization."""
    times = {}
    start = perf_counter()
    A = canonicalize(A)
    csc = CSC.from_csr(A)
    times["input_csc"] = perf_counter() - start
    if trace:
        print(f"1. canonical input / CSC: n={A.shape[0]}, nnz={len(A.data)}")
    start = perf_counter()
    scaled, scaling = equilibrate(A, equilibration)
    times["equilibration"] = perf_counter() - start
    if trace:
        print(f"2. equilibration: equed={scaling.equed}")
    reused = analysis is not None
    if analysis is None:
        analysis = analyze(A, ordering=ordering, postorder=postorder,
                           panel_size=panel_size, relax=relax)
        times["column_ordering"] = analysis.ordering_seconds
        times["symbolic_analysis"] = analysis.symbolic_seconds
    else:
        analysis.check_pattern(A)
        times["column_ordering"] = times["symbolic_analysis"] = 0.0
    if trace:
        print(f"3. column ordering: {analysis.ordering}, reused={reused}")
        print(f"4. etree / postorder: postordered={analysis.postordered}, "
              f"roots={sum(parent == A.shape[0] for parent in analysis.etree)}")
        print(f"5. structural analysis: envelope nnz={analysis.envelope_nnz}, "
              f"strict groups={len(analysis.strict_groups)}, relaxed groups={len(analysis.relaxed_groups)}")
        print(f"6. panel scheduling: {len(analysis.panels)} panels")
    start = perf_counter()
    lower, upper, rows, stats = factor_panels(
        scaled, analysis, pivot_threshold=pivot_threshold, pivot_atol=pivot_atol,
        preferred_row_order=preferred_row_order, trace=trace,
    )
    times["numerical_factorization"] = perf_counter() - start
    start = perf_counter()
    nodes = pack_supernodes(lower, upper, analysis.panel_size)
    upper_csr = CSR.from_coo(A.shape, ((i, j, value)
                                      for i, row in enumerate(upper) for j, value in row.items()))
    upper_csc = CSC.from_csr(upper_csr)
    times["supernode_packing"] = perf_counter() - start
    if trace:
        print(f"7. threshold-pivoted LU: row swaps={stats.row_swaps}, fill insertions={stats.fill_insertions}")
        print(f"8. actual supernode packing: {len(nodes)} blocks")
    return SparseDirectSolver(A, csc, scaled, scaling, analysis, lower, upper, rows,
                              nodes, upper_csc, stats, equilibration, pivot_threshold, pivot_atol,
                              reused, times)


def demo() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("poisson", "pivot", "scaled"), default="poisson")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--ordering", choices=("natural", "minimum_degree"), default="minimum_degree")
    parser.add_argument("--no-postorder", action="store_true")
    parser.add_argument("--panel-size", type=int, default=4)
    parser.add_argument("--relax", type=int, default=4)
    parser.add_argument("--equilibration", choices=("auto", "both", "none"), default="auto")
    parser.add_argument("--pivot-threshold", type=float, default=1.0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--compare-scipy", action="store_true")
    args = parser.parse_args()
    if args.grid < 1 or args.panel_size < 1 or args.relax < 1:
        parser.error("grid, panel-size, and relax must be positive")
    if not isfinite(args.pivot_threshold) or not 0 <= args.pivot_threshold <= 1:
        parser.error("pivot-threshold must be in [0,1]")
    A = pivoting_example() if args.example == "pivot" else poisson_2d(args.grid)
    n = A.shape[0]
    if args.example == "scaled":
        r = [10.0 ** (6 * i / max(1, n - 1)) for i in range(n)]
        c = [10.0 ** (-4 * i / max(1, n - 1)) for i in range(n)]
        A = CSR(A.shape, tuple(A.data[k] * r[i] * c[A.indices[k]]
                              for i in range(n) for k in range(A.indptr[i], A.indptr[i + 1])),
                A.indices, A.indptr)
    expected = [1.0 + sin(0.13 * i) for i in range(n)]
    b = A.matvec(expected)
    solver = factorize(A, ordering=args.ordering, postorder=not args.no_postorder,
                       panel_size=args.panel_size, relax=args.relax,
                       equilibration=args.equilibration, pivot_threshold=args.pivot_threshold,
                       trace=args.trace)
    report = solver.refine(b)
    error = norm([xi - ei for xi, ei in zip(report.x, expected)]) / norm(expected)
    print(f"9. supernodal triangular solves: relative solution error={error:.3e}")
    print(f"10. refinement: steps={report.refinement_steps}, componentwise berr={report.componentwise_backward_error:.3e}")
    print(f"11. diagnostics: residual={report.relative_residual:.3e}, "
          f"rcond_1(original)~{report.reciprocal_condition_estimate:.3e}, "
          f"ferr~{report.forward_error_estimate:.3e}, reciprocal pivot growth={report.reciprocal_pivot_growth:.3e}")
    print(f"12. storage: nnz(A)={len(A.data)}, nnz(L)={solver.nnz_L}, nnz(U)={solver.nnz_U}, "
          f"packed dense values={solver.packed_values}")
    print("    (L count includes its unit diagonal; packed blocks include stored zeros.)")
    print(f"    U is also available in CSC; peak working sparse entries={solver.numeric_stats.peak_work_nnz}")
    second = solver.solve_many([b, A.matvec([1.0] * n)])
    second_error = norm([xi - 1.0 for xi in second[1]]) / (n ** 0.5)
    print(f"13. factor reuse / multiple RHS: second solution error={second_error:.3e}")
    changed = CSR(A.shape, tuple(value * 1.1 for value in A.data), A.indices, A.indptr)
    reused = solver.refactor(changed)
    changed_x = reused.solve(changed.matvec(expected))
    changed_error = norm([xi - ei for xi, ei in zip(changed_x, expected)]) / norm(expected)
    print(f"14. same-pattern refactor: symbolic reused={reused.symbolic_reused}, "
          f"accepted old pivots={reused.numeric_stats.reused_pivots}, error={changed_error:.3e}")
    print("Stage timings (seconds; solve time also includes diagnostic solves):")
    for stage, seconds in solver.stage_seconds.items():
        print(f"    {stage}: {seconds:.6f}")
    if args.compare_scipy:
        try:
            import numpy as np
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import splu
        except ImportError:
            parser.error("--compare-scipy requires optional numpy and scipy packages")
        csc = solver.csc
        reference_A = csc_matrix((csc.data, csc.row_indices, csc.col_ptr), shape=A.shape)
        reference = splu(reference_A).solve(np.asarray(b))
        difference = norm([xi - ri for xi, ri in zip(report.x, reference)]) / norm(reference)
        print(f"SciPy / SuperLU comparison: relative solution difference={difference:.3e}")


if __name__ == "__main__":
    demo()
