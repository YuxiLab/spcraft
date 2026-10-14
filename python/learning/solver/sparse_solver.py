"""Learn sparse linear solving from scratch, using only Python's standard library.

Read in this order: CSR.from_coo -> CSR.matvec -> conjugate_gradient -> demo.
CG requires a real symmetric positive-definite (SPD) matrix. The curvature
checks below detect some invalid inputs; they do not certify that a matrix is SPD.

Run: python3 sparse_learning/sparse_solver.py --grid 16
     python3 sparse_learning/sparse_solver.py --grid 16 --scaled
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import fsum, hypot, isfinite, sin
from time import perf_counter
from typing import Iterable, Sequence


def dot(x: Sequence[float], y: Sequence[float]) -> float:
    return fsum(a * b for a, b in zip(x, y))


def norm(x: Sequence[float]) -> float:
    return hypot(*x)


@dataclass(frozen=True)
class CSR:
    """Compressed sparse rows: row i occupies indptr[i]:indptr[i+1]."""

    shape: tuple[int, int]
    data: tuple[float, ...]
    indices: tuple[int, ...]
    indptr: tuple[int, ...]

    def __post_init__(self) -> None:
        rows, cols = self.shape
        if rows < 0 or cols < 0:
            raise ValueError("Matrix dimensions must be nonnegative")
        if len(self.indptr) != rows + 1 or self.indptr[0] != 0:
            raise ValueError("indptr must have rows+1 entries and start at zero")
        if len(self.data) != len(self.indices) or self.indptr[-1] != len(self.data):
            raise ValueError("CSR array lengths are inconsistent")
        if any(a > b for a, b in zip(self.indptr, self.indptr[1:])):
            raise ValueError("indptr must be nondecreasing")
        if any(j < 0 or j >= cols for j in self.indices):
            raise ValueError("Column index out of bounds")
        if not all(isfinite(v) for v in self.data):
            raise ValueError("Matrix entries must be finite")

    @classmethod
    def from_coo(
        cls, shape: tuple[int, int], entries: Iterable[tuple[int, int, float]]
    ) -> CSR:
        """Convert (row, column, value) triplets; sum duplicates and drop zeros."""
        rows, cols = shape
        if rows < 0 or cols < 0:
            raise ValueError("Matrix dimensions must be nonnegative")
        buckets: list[dict[int, float]] = [{} for _ in range(rows)]
        for i, j, value in entries:
            if not (0 <= i < rows and 0 <= j < cols):
                raise ValueError("Triplet index out of bounds")
            value = float(value)
            if not isfinite(value):
                raise ValueError("Matrix entries must be finite")
            buckets[i][j] = buckets[i].get(j, 0.0) + value
        data, indices, indptr = [], [], [0]
        for row in buckets:
            for j, value in sorted(row.items()):
                if value != 0.0:
                    indices.append(j)
                    data.append(value)
            indptr.append(len(data))
        return cls(shape, tuple(data), tuple(indices), tuple(indptr))

    def matvec(self, x: Sequence[float]) -> list[float]:
        """Compute A*x in O(nnz + rows) work without constructing a dense A."""
        if len(x) != self.shape[1]:
            raise ValueError("Vector length does not match matrix columns")
        return [
            fsum(self.data[k] * x[self.indices[k]]
                 for k in range(self.indptr[i], self.indptr[i + 1]))
            for i in range(self.shape[0])
        ]

    def diagonal(self) -> list[float]:
        return [
            fsum(self.data[k] for k in range(self.indptr[i], self.indptr[i + 1])
                 if self.indices[k] == i)
            for i in range(self.shape[0])
        ]


@dataclass
class CGResult:
    x: list[float]
    converged: bool
    iterations: int
    residual_norms: list[float]  # Actual ||b - A*x||, including the initial guess.


def conjugate_gradient(
    A: CSR,
    b: Sequence[float],
    *,
    x0: Sequence[float] | None = None,
    jacobi: bool = False,
    rtol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> CGResult:
    """Solve SPD A*x=b with CG or Jacobi-preconditioned CG.

    Stop when ||b-A*x|| <= max(atol, rtol*||b||). Jacobi applies the inverse
    diagonal to each residual: z_i = r_i / A_ii. It never forms an inverse.

    For teaching, we explicitly recompute the true residual every iteration.
    This adds one SpMV per iteration to the usual CG cost, but keeps the
    convergence history honest in floating-point arithmetic.
    """
    n, m = A.shape
    if n != m or len(b) != n:
        raise ValueError("CG requires a square matrix and a matching RHS")
    if not all(isfinite(t) and t >= 0 for t in (rtol, atol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if maxiter is None:
        maxiter = max(1, 10 * n)
    if not isinstance(maxiter, int) or maxiter < 0:
        raise ValueError("maxiter must be a nonnegative integer")
    b = [float(v) for v in b]
    x = [0.0] * n if x0 is None else [float(v) for v in x0]
    if len(x) != n or not all(isfinite(v) for v in b + x):
        raise ValueError("RHS and initial guess must be finite, matching vectors")

    diagonal = A.diagonal() if jacobi else [1.0] * n
    if any(d <= 0 or not isfinite(d) for d in diagonal):
        raise ValueError("Jacobi-preconditioned CG requires a positive diagonal")

    r = [bi - ai for bi, ai in zip(b, A.matvec(x))]
    history = [norm(r)]
    tolerance = max(atol, rtol * norm(b))
    if history[-1] <= tolerance:
        return CGResult(x, True, 0, history)

    # z = M^{-1}r; with no preconditioner, M is the identity and z=r.
    z = [ri / d for ri, d in zip(r, diagonal)]
    p = z.copy()
    rho = dot(r, z)

    for iteration in range(1, maxiter + 1):
        Ap = A.matvec(p)
        curvature = dot(p, Ap)
        if curvature <= 0 or rho <= 0 or not all(
            isfinite(v) for v in (curvature, rho)
        ):
            raise ValueError("CG breakdown: invalid curvature or numerical overflow")

        # Choose the step, update the solution and recursive residual.
        alpha = rho / curvature
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * api for ri, api in zip(r, Ap)]

        true_r = [bi - ai for bi, ai in zip(b, A.matvec(x))]
        history.append(norm(true_r))
        if not isfinite(history[-1]):
            raise ValueError("CG produced a nonfinite residual")
        if history[-1] <= tolerance:
            return CGResult(x, True, iteration, history)

        # If the recursive residual falsely suggests convergence, restart
        # from the actual residual rather than declaring success.
        restart = norm(r) <= tolerance
        if restart:
            r = true_r
        z = [ri / d for ri, d in zip(r, diagonal)]
        rho_new = dot(r, z)
        beta = rho_new / rho
        p = z.copy() if restart else [zi + beta * pi for zi, pi in zip(z, p)]
        rho = rho_new

    return CGResult(x, False, maxiter, history)


def poisson_2d(grid: int) -> CSR:
    """Five-point negative Laplacian on a grid with zero Dirichlet boundaries.

    Each interior unknown has diagonal 4 and up to four neighbors of value -1.
    Keeping diagonal 4 at boundary-adjacent points represents prescribed zero
    values outside the interior grid. This matrix is SPD.
    """
    if grid < 1:
        raise ValueError("grid must be positive")
    entries = []
    for row in range(grid):
        for col in range(grid):
            i = row * grid + col
            entries.append((i, i, 4.0))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < grid and 0 <= cc < grid:
                    entries.append((i, rr * grid + cc, -1.0))
    return CSR.from_coo((grid * grid, grid * grid), entries)


def demo() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--scaled", action="store_true",
                        help="Use D*A*D with varying diagonal D; still SPD")
    args = parser.parse_args()
    if args.grid < 1:
        parser.error("--grid must be positive")
    A = poisson_2d(args.grid)
    n = A.shape[0]
    if args.scaled:
        # Congruence by a positive diagonal preserves SPD but worsens scaling.
        d = [10.0 ** (2.0 * i / max(1, n - 1)) for i in range(n)]
        scaled_data = tuple(
            A.data[k] * d[i] * d[A.indices[k]]
            for i in range(n)
            for k in range(A.indptr[i], A.indptr[i + 1])
        )
        A = CSR(A.shape, scaled_data, A.indices, A.indptr)

    # Manufacture a known solution so both residual and solution error can
    # be measured. A small residual alone does not guarantee a small error.
    expected = [1.0 + sin(0.13 * i) for i in range(n)]
    b = A.matvec(expected)
    print(f"unknowns={n}, nonzeros={len(A.data)}, scaled={args.scaled}")
    successful = True
    for jacobi in (False, True):
        start = perf_counter()
        result = conjugate_gradient(A, b, jacobi=jacobi, maxiter=10 * n)
        elapsed = perf_counter() - start
        error = norm([xi - ei for xi, ei in zip(result.x, expected)]) / norm(expected)
        relative_residual = result.residual_norms[-1] / norm(b)
        label = "CG + Jacobi" if jacobi else "CG"
        print(f"{label:12s}: converged={result.converged}, "
              f"iterations={result.iterations}, residual={relative_residual:.3e}, "
              f"error={error:.3e}, seconds={elapsed:.4f}")
        successful &= result.converged
    if not args.scaled:
        print("Here diag(A)=4I, so Jacobi only rescales CG. "
              "Use --scaled to see a useful preconditioning example.")
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    demo()
