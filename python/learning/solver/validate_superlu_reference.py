"""Optional independent check against SciPy's actual SuperLU factorization.

Requires numpy/scipy only for this validation script, not for the solver.
Run: python3 sparse_learning/validate_superlu_reference.py
Dense arrays here are small reference checks; the implementation stays sparse.
"""

from __future__ import annotations

from itertools import product

if __package__:
    from .sparse_solver import CSR, poisson_2d
    from .sparse_lu import pivoting_example
    from .superlu_pipeline import factorize
else:
    from sparse_solver import CSR, poisson_2d
    from sparse_lu import pivoting_example
    from superlu_pipeline import factorize


def main() -> None:
    try:
        import numpy as np
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import splu
    except ImportError as error:
        raise SystemExit("This optional reference check requires numpy and scipy") from error

    rng = np.random.default_rng(20261012)
    examples = [pivoting_example()] + [poisson_2d(grid) for grid in (1, 3, 5, 8)]
    for n in (3, 6, 12, 20):
        for _ in range(2):
            dense = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.25)
            # Small diagonals permit plenty of off-diagonal pivoting, while
            # preserving structural nonsingularity. Values are nonsymmetric.
            dense += np.diag(rng.uniform(0.1, 0.3, size=n))
            shuffled = rng.permutation(n)
            dense = dense[shuffled]
            examples.append(CSR.from_coo((n, n), ((i, j, float(dense[i, j]))
                                                  for i, j in zip(*np.nonzero(dense)))))

    cases, worst_solve, worst_factor, worst_rcond_ratio = 0, 0.0, 0.0, 0.0
    for A in examples:
        dense = np.zeros(A.shape)
        for i in range(A.shape[0]):
            for k in range(A.indptr[i], A.indptr[i + 1]):
                dense[i, A.indices[k]] = A.data[k]
        reference = splu(csc_matrix(dense))
        rhs = rng.normal(size=A.shape[0])
        exact_rcond = 1.0 / (np.linalg.norm(dense, 1) * np.linalg.norm(np.linalg.inv(dense), 1))
        for ordering, width, equilibration in product(
            ("natural", "minimum_degree"), (1, 3, 8), ("none", "both")
        ):
            solver = factorize(A, ordering=ordering, panel_size=width,
                               equilibration=equilibration)
            for transpose in (False, True):
                actual = np.asarray(solver.solve(rhs, transpose=transpose))
                expected = reference.solve(rhs, trans="T" if transpose else "N")
                relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                worst_solve = max(worst_solve, relative)
                if relative > 1e-8:
                    raise AssertionError(f"SuperLU solution mismatch: {relative}, "
                                         f"n={solver.n}, width={width}, trans={transpose}")
            lower, upper = np.eye(solver.n), np.zeros(A.shape)
            for i, row in enumerate(solver.L):
                for j, value in row.items():
                    lower[i, j] = value
            for i, row in enumerate(solver.U):
                for j, value in row.items():
                    upper[i, j] = value
            scaled = dense * np.asarray(solver.scaling.row)[:, None]
            scaled *= np.asarray(solver.scaling.column)[None, :]
            permuted = scaled[np.ix_(solver.row_order, solver.analysis.col_order)]
            factor_error = np.linalg.norm(lower @ upper - permuted, np.inf) / np.linalg.norm(permuted, np.inf)
            worst_factor = max(worst_factor, factor_error)
            if factor_error > 1e-11:
                raise AssertionError(f"P_r*R*A*C*P_c != L*U: {factor_error}")
            estimated_rcond = solver.reciprocal_condition_estimate()
            if estimated_rcond < exact_rcond * (1 - 1e-8):
                raise AssertionError("Inverse norm estimate exceeded the exact norm")
            worst_rcond_ratio = max(worst_rcond_ratio, estimated_rcond / exact_rcond)
            cases += 1
    print(f"PASS: {cases} configurations, {2 * cases} normal/transpose SuperLU comparisons")
    print(f"Worst relative solution difference: {worst_solve:.3e}")
    print(f"Worst relative factorization residual: {worst_factor:.3e}")
    print(f"Largest estimated/exact reciprocal condition ratio: {worst_rcond_ratio:.3f}")


if __name__ == "__main__":
    main()
