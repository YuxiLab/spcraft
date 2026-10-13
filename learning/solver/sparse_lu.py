"""Sparse direct LU, from scratch, with only the Python standard library.

Study order: sparse_lu -> LUFactors.solve -> demo.

    P A = L U        (L has an implicit unit diagonal)
    L y = P b        (forward substitution)
    U x = y          (back substitution)

Rows are dictionaries {column: value}. Unlike fixed CSR arrays, dictionaries
can accept fill-in: new nonzeros created by elimination. No dense matrix is
allocated. Numerical zeros are removed exactly; small nonzeros are not dropped.

This first implementation uses natural column ordering and scalar elimination.
It scans remaining rows for each pivot, so scanning alone costs O(n^2).
It teaches numerical factorization; symbolic analysis, fill-reducing ordering,
and supernodes are subsequent topics needed for a scalable implementation.

Run the small pivoting example: python3 sparse_learning/sparse_lu.py --trace
Run a Poisson example:         python3 sparse_learning/sparse_lu.py --grid 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import fsum, isfinite, sin
from time import perf_counter
from typing import Sequence

if __package__:
    from .sparse_solver import CSR, norm, poisson_2d
else:
    from sparse_solver import CSR, norm, poisson_2d


@dataclass
class LUFactors:
    # L stores only entries below the diagonal; its diagonal is implicitly 1.
    L: list[dict[int, float]]
    U: list[dict[int, float]]
    # perm[i] is the original row now occupying factor row i.
    # Thus (P*b)[i] = b[perm[i]].
    perm: list[int]
    input_nnz: int

    @property
    def nnz_L(self) -> int:
        return len(self.L) + sum(len(row) for row in self.L)

    @property
    def nnz_U(self) -> int:
        return sum(len(row) for row in self.U)

    def solve(self, b: Sequence[float]) -> list[float]:
        """Reuse these factors for a new RHS, without repeating elimination."""
        n = len(self.perm)
        if len(b) != n:
            raise ValueError("RHS length must match the factors")
        b = [float(value) for value in b]
        if not all(isfinite(value) for value in b):
            raise ValueError("RHS must be finite")

        # Forward solve L*y = P*b. No division: L[i,i] = 1.
        y = [0.0] * n
        for i in range(n):
            y[i] = b[self.perm[i]] - fsum(
                lij * y[j] for j, lij in self.L[i].items()
            )

        # Back solve U*x = y. Columns were not permuted, so x is already
        # expressed in the original variable order.
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            off_diagonal = fsum(
                uij * x[j] for j, uij in self.U[i].items() if j > i
            )
            x[i] = (y[i] - off_diagonal) / self.U[i][i]
        if not all(isfinite(value) for value in x):
            raise ArithmeticError("Nonfinite solution; numerical overflow")
        return x


def sparse_lu(A: CSR, *, pivot_atol: float = 0.0, trace: bool = False) -> LUFactors:
    """Factor a real square sparse matrix with partial row pivoting: P*A=L*U.

    At step k, choose the largest |U[i,k]| among remaining rows. Store the
    multipliers in L and update only entries present in the pivot row.

    pivot_atol is an absolute rejection threshold. Its default rejects only
    zero pivots; a nonzero threshold depends on the matrix's scale and is not
    a condition-number estimate. No global SPD or symmetry assumption is made.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("LU requires a square matrix")
    if not isfinite(pivot_atol) or pivot_atol < 0:
        raise ValueError("pivot_atol must be finite and nonnegative")

    # Copy the input; factorization must not overwrite the user's A.
    # Sum duplicates even if A was constructed directly rather than from_coo.
    U: list[dict[int, float]] = []
    for i in range(n):
        row: dict[int, float] = {}
        for k in range(A.indptr[i], A.indptr[i + 1]):
            j = A.indices[k]
            row[j] = row.get(j, 0.0) + A.data[k]
        if not all(isfinite(value) for value in row.values()):
            raise ArithmeticError("Overflow while summing input entries")
        U.append({j: value for j, value in row.items() if value != 0.0})
    L: list[dict[int, float]] = [{} for _ in range(n)]
    perm = list(range(n))
    input_nnz = sum(len(row) for row in U)

    for k in range(n):
        # Partial pivoting controls multiplier magnitudes in this column.
        pivot_row = max(range(k, n), key=lambda i: abs(U[i].get(k, 0.0)))
        pivot = U[pivot_row].get(k, 0.0)
        if abs(pivot) <= pivot_atol:
            raise ValueError(f"Zero or rejected pivot at column {k}: {pivot}")

        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            perm[k], perm[pivot_row] = perm[pivot_row], perm[k]
            # Crucial: also swap the multipliers from PREVIOUS steps.
            # Otherwise, late pivoting breaks P*A = L*U.
            L[k], L[pivot_row] = L[pivot_row], L[k]
            if trace:
                print(f"k={k}: swap rows {k} and {pivot_row}, including old L entries")
        if trace:
            print(f"k={k}: pivot U[{k},{k}] = {pivot:g}")

        # Only these pivot-row nonzeros can change the remaining rows.
        pivot_tail = [(j, value) for j, value in U[k].items() if j > k]
        for i in range(k + 1, n):
            aik = U[i].pop(k, 0.0)
            if aik == 0.0:
                continue
            multiplier = aik / pivot
            L[i][k] = multiplier
            if trace:
                print(f"  L[{i},{k}] = {aik:g} / {pivot:g} = {multiplier:g}")

            for j, ukj in pivot_tail:
                old = U[i].get(j, 0.0)
                new = old - multiplier * ukj
                if not isfinite(new):
                    raise ArithmeticError(f"Overflow updating entry ({i}, {j})")
                if new == 0.0:
                    U[i].pop(j, None)
                else:
                    U[i][j] = new
                if trace:
                    fill = " [fill-in]" if old == 0.0 and new != 0.0 else ""
                    print(f"  U[{i},{j}]: {old:g} - "
                          f"({multiplier:g})*({ukj:g}) = {new:g}{fill}")

    return LUFactors(L, U, perm, input_nnz)


def pivoting_example() -> CSR:
    """A nonsymmetric example with both an initial and a late row swap."""
    # A = [0 2 1; 1 1 0; 2 0 3]. With x=[1,2,3], b=[7,3,11].
    return CSR.from_coo((3, 3), [
        (0, 1, 2), (0, 2, 1),
        (1, 0, 1), (1, 1, 1),
        (2, 0, 2), (2, 2, 3),
    ])


def demo() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, help="Use a grid x grid Poisson problem")
    parser.add_argument("--trace", action="store_true", help="Print elimination steps")
    args = parser.parse_args()
    if args.grid is not None and args.grid < 1:
        parser.error("--grid must be positive")
    A = pivoting_example() if args.grid is None else poisson_2d(args.grid)
    n = A.shape[0]
    expected = ([1.0, 2.0, 3.0] if args.grid is None
                else [1.0 + sin(0.13 * i) for i in range(n)])
    b = A.matvec(expected)

    start = perf_counter()
    factors = sparse_lu(A, trace=args.trace)
    factor_time = perf_counter() - start
    start = perf_counter()
    x = factors.solve(b)
    solve_time = perf_counter() - start

    print(f"unknowns={n}, nnz(A)={factors.input_nnz}, "
          f"nnz(L)={factors.nnz_L} (including unit diagonal), nnz(U)={factors.nnz_U}")
    # Count the diagonal once when comparing combined factor storage with A.
    storage_ratio = (factors.nnz_L + factors.nnz_U - n) / factors.input_nnz
    print(f"combined factor nonzeros / input nonzeros = {storage_ratio:.3f}")
    print(f"factorization={factor_time:.6f}s, triangular solves={solve_time:.6f}s")
    residual = norm([bi - ai for bi, ai in zip(b, A.matvec(x))]) / norm(b)
    error = norm([xi - ei for xi, ei in zip(x, expected)]) / norm(expected)
    print(f"relative residual={residual:.3e}, relative solution error={error:.3e}")
    if n <= 10:
        print(f"row permutation={factors.perm}")
        print(f"x={x}")

    # A second RHS reuses the factorization. The solve routine does not change L/U.
    second_expected = [1.0] * n
    second_x = factors.solve(A.matvec(second_expected))
    second_error = norm([xi - 1.0 for xi in second_x]) / norm(second_expected)
    print(f"second RHS, same factors: relative solution error={second_error:.3e}")


if __name__ == "__main__":
    demo()
