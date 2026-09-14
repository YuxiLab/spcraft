#!/usr/bin/env python3
"""Rank a directed graph with SPCraft PageRank and visualize the result."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import scipy.sparse as sp

import spcraft


EDGES = (
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 0),
    (2, 6),
    (3, 0),
    (3, 1),
    (3, 4),
    (4, 5),
    (5, 0),
    (5, 4),
)
VERTICES = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--damping", type=float, default=0.85, help="Link-following probability."
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-12, help="L1 convergence target."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=500, help="PageRank iteration cap."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pagerank_graph.png"),
        help="Figure filename.",
    )
    return parser.parse_args()


def draw_graph(axis: plt.Axes, ranks: np.ndarray) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, VERTICES, endpoint=False) + np.pi / 2.0
    positions = np.column_stack((np.cos(angles), np.sin(angles)))

    for source, target in EDGES:
        arrow = FancyArrowPatch(
            positions[source],
            positions[target],
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.12",
            mutation_scale=11,
            color="#667580",
            linewidth=1.0,
            shrinkA=16,
            shrinkB=16,
            alpha=0.8,
        )
        axis.add_patch(arrow)

    node_sizes = 900.0 + 5_500.0 * ranks
    nodes = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=node_sizes,
        c=ranks,
        cmap="YlOrRd",
        edgecolors="#202A34",
        linewidths=1.2,
        zorder=3,
    )
    for vertex, (horizontal, vertical) in enumerate(positions):
        axis.text(
            horizontal,
            vertical,
            str(vertex),
            ha="center",
            va="center",
            fontweight="bold",
            zorder=4,
        )
    axis.set_title("node size and color encode PageRank")
    axis.set_aspect("equal")
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.35, 1.35)
    axis.axis("off")
    plt.colorbar(nodes, ax=axis, shrink=0.72, label="rank")


def main() -> None:
    arguments = parse_args()
    rows, columns = np.asarray(EDGES, dtype=np.int32).T
    adjacency = sp.csr_array(
        (np.ones(len(EDGES)), (rows, columns)), shape=(VERTICES, VERTICES)
    )
    matrix = spcraft.from_scipy(adjacency)
    ranks, info = matrix.pagerank(
        damping=arguments.damping,
        tolerance=arguments.tolerance,
        max_iterations=arguments.max_iterations,
    )
    if not info["converged"]:
        raise RuntimeError(f"PageRank failed to converge: {info}")
    np.testing.assert_allclose(ranks.sum(), 1.0, atol=1e-12)

    order = np.argsort(ranks)[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    draw_graph(axes[0], ranks)
    axes[1].bar(
        np.arange(VERTICES), ranks[order], color="#990000", edgecolor="#202A34"
    )
    axes[1].set_xticks(np.arange(VERTICES), [str(vertex) for vertex in order])
    axes[1].set_xlabel("page, ordered by rank")
    axes[1].set_ylabel("stationary probability")
    axes[1].set_title("PageRank distribution")
    axes[1].grid(axis="y", alpha=0.25)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    plt.close(figure)

    print(
        f"converged in {info['iterations']} iterations; "
        f"final L1 change {info['residual']:.3e}"
    )
    for position, vertex in enumerate(order, start=1):
        print(f"{position:2d}. page {vertex}: {ranks[vertex]:.6f}")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
