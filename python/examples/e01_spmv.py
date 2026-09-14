#!/usr/bin/env python3
"""Run SPCraft SpMV and visualize the sparse operator and its vectors."""

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
        "--size", type=int, default=80, help="Number of rows and columns."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spmv_example.png"),
        help="Figure filename.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.size < 3:
        raise ValueError("--size must be at least 3")

    size = arguments.size
    # A tridiagonal second-difference operator has only three populated
    # diagonals, yet its action on a signal is easy to recognize visually.
    off_diagonal = -np.ones(size - 1)
    scipy_matrix = sp.diags(
        (off_diagonal, 2.0 * np.ones(size), off_diagonal),
        offsets=(-1, 0, 1),
        format="csr",
    )
    matrix = spcraft.from_scipy(scipy_matrix)

    coordinates = np.linspace(0.0, 1.0, size)
    x = np.sin(2.0 * np.pi * coordinates) + 0.25 * np.sin(
        12.0 * np.pi * coordinates
    )
    y = matrix @ x
    reference = scipy_matrix @ x
    np.testing.assert_allclose(y, reference, rtol=1e-12, atol=1e-12)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].spy(scipy_matrix, markersize=3, color="#990000")
    axes[0].set_title(f"CSR pattern ({matrix.nnz} nonzeros)")
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("row")

    axes[1].plot(coordinates, x, color="#006298", linewidth=2)
    axes[1].set_title("input vector $x$")
    axes[1].set_xlabel("coordinate")
    axes[1].grid(alpha=0.25)

    axes[2].plot(coordinates, y, color="#990000", linewidth=2)
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].set_title("SPCraft result $y=Ax$")
    axes[2].set_xlabel("coordinate")
    axes[2].grid(alpha=0.25)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    plt.close(figure)

    print(f"matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    print(f"maximum error against SciPy: {np.max(np.abs(y - reference)):.3e}")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
