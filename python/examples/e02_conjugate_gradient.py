#!/usr/bin/env python3
"""Solve a sparse 2D Poisson system with SPCraft conjugate gradient."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import spcraft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=int, default=32, help="Grid points along one dimension."
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-10, help="Relative residual target."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=1_000, help="CG iteration cap."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cg_solution.png"),
        help="Figure filename.",
    )
    return parser.parse_args()


def laplacian_2d(grid: int) -> sp.csr_array:
    """Return the full five-point negative-Laplacian stencil in CSR."""
    off_diagonal = -np.ones(grid - 1)
    one_dimensional = sp.diags_array(
        (off_diagonal, 2.0 * np.ones(grid), off_diagonal), offsets=(-1, 0, 1)
    )
    identity = sp.eye_array(grid, format="csr")
    return sp.csr_array(
        sp.kron(identity, one_dimensional) + sp.kron(one_dimensional, identity)
    )


def main() -> None:
    arguments = parse_args()
    if arguments.grid < 2:
        raise ValueError("--grid must be at least 2")

    scipy_matrix = laplacian_2d(arguments.grid)
    matrix = spcraft.from_scipy(scipy_matrix)

    coordinates = np.linspace(0.0, 1.0, arguments.grid + 2)[1:-1]
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    exact_field = (
        horizontal
        * (1.0 - horizontal)
        * vertical
        * (1.0 - vertical)
        * np.exp(0.5 * horizontal + 0.25 * vertical)
    )
    exact = np.ascontiguousarray(exact_field.ravel())

    # Forming b through SPCraft makes the complete example a round trip:
    # one explicit SpMV followed by the repeated SpMVs inside CG.
    rhs = matrix @ exact
    solution, info = matrix.solve_cg(
        rhs,
        tolerance=arguments.tolerance,
        max_iterations=arguments.max_iterations,
    )
    if not info["converged"] or info["breakdown"]:
        raise RuntimeError(f"CG failed: {info}")

    computed_field = solution.reshape((arguments.grid, arguments.grid))
    absolute_error = np.abs(computed_field - exact_field)
    relative_error = np.linalg.norm(solution - exact) / np.linalg.norm(exact)

    figure, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    axes[0].spy(scipy_matrix, markersize=0.45, color="#990000")
    axes[0].set_title(f"five-point matrix $A$\n({matrix.nnz} nonzeros)")
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("row")

    common = {"origin": "lower", "cmap": "viridis"}
    exact_image = axes[1].imshow(exact_field, **common)
    axes[1].set_title("known solution")
    figure.colorbar(exact_image, ax=axes[1], shrink=0.78)

    computed_image = axes[2].imshow(computed_field, **common)
    axes[2].set_title(f"CG solution ({info['iterations']} iterations)")
    figure.colorbar(computed_image, ax=axes[2], shrink=0.78)

    error_floor = np.finfo(absolute_error.dtype).eps
    error_image = axes[3].imshow(
        np.log10(absolute_error + error_floor), origin="lower", cmap="magma"
    )
    axes[3].set_title(r"$\log_{10}$ absolute error")
    figure.colorbar(error_image, ax=axes[3], shrink=0.78)

    for axis in axes[1:]:
        axis.set_xlabel("grid column")
        axis.set_ylabel("grid row")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    plt.close(figure)

    print(f"matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    print(
        f"iterations: {info['iterations']}, relative residual: {info['residual']:.3e}"
    )
    print(f"relative solution error: {relative_error:.3e}")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
