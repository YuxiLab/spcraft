#!/usr/bin/env python3
"""Generate deterministic CFD and graph sparsity figures for Beamer slides.

The examples are synthetic teaching matrices.  They preserve the structural
features of the application families without depending on external datasets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
from scipy.sparse import coo_array, csr_array, diags, eye, kron
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "beamer" / "assets" / "generated"

INK = "#202A34"
SOFT = "#667580"
RED = "#990000"
ORANGE = "#DF6C00"
PALE_RED = "#F4E6E6"
PALE_ORANGE = "#FAEBDD"


def structured_cfd(nx: int, ny: int) -> csr_array:
    """Nonsymmetric five-point convection--diffusion operator."""
    n = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for j in range(ny):
        for i in range(nx):
            p = j * nx + i
            rows.append(p)
            cols.append(p)
            vals.append(4.2)
            for di, dj, value in [(-1, 0, -1.3), (1, 0, -0.7), (0, -1, -1.1), (0, 1, -0.9)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    rows.append(p)
                    cols.append(jj * nx + ii)
                    vals.append(value)
    return coo_array((vals, (rows, cols)), shape=(n, n)).tocsr()


def block_cfd(nx: int, ny: int, variables: int = 3) -> csr_array:
    """Cell-centered multiphysics operator with small dense variable blocks."""
    scalar = structured_cfd(nx, ny)
    local = np.ones((variables, variables), dtype=float)
    local[np.diag_indices(variables)] = 2.0
    return (kron(scalar, local, format="csr") + kron(eye(scalar.shape[0]), local, format="csr")).tocsr()


def unstructured_mesh(n_points: int = 210, seed: int = 7) -> tuple[np.ndarray, Delaunay, csr_array]:
    """Delaunay mesh and a finite-element-style node adjacency matrix."""
    rng = np.random.default_rng(seed)
    interior = rng.random((n_points, 2))
    boundary_t = np.linspace(0.0, 1.0, 18)
    boundary = np.vstack(
        [
            np.column_stack([boundary_t, np.zeros_like(boundary_t)]),
            np.column_stack([boundary_t, np.ones_like(boundary_t)]),
            np.column_stack([np.zeros_like(boundary_t[1:-1]), boundary_t[1:-1]]),
            np.column_stack([np.ones_like(boundary_t[1:-1]), boundary_t[1:-1]]),
        ]
    )
    points = np.vstack([interior, boundary])
    tri = Delaunay(points)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            u, v = sorted((int(simplex[a]), int(simplex[b])))
            edges.add((u, v))
    rows = list(range(len(points)))
    cols = list(range(len(points)))
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])
    vals = np.ones(len(rows), dtype=float)
    adjacency = coo_array((vals, (rows, cols)), shape=(len(points), len(points))).tocsr()
    return points, tri, adjacency


def grid_graph(nx: int, ny: int) -> csr_array:
    """Road-like planar graph with a few deterministic diagonal connectors."""
    rows: list[int] = []
    cols: list[int] = []
    for j in range(ny):
        for i in range(nx):
            p = j * nx + i
            for di, dj in [(1, 0), (0, 1)]:
                ii, jj = i + di, j + dj
                if ii < nx and jj < ny:
                    q = jj * nx + ii
                    rows.extend([p, q])
                    cols.extend([q, p])
            if (i + 2 * j) % 7 == 0 and i + 1 < nx and j + 1 < ny:
                q = (j + 1) * nx + i + 1
                rows.extend([p, q])
                cols.extend([q, p])
    vals = np.ones(len(rows))
    return coo_array((vals, (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()


def small_world_graph(n: int = 360, k: int = 4, shortcuts: int = 170, seed: int = 12) -> csr_array:
    """Ring-lattice graph with random long-range shortcuts."""
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    for u in range(n):
        for d in range(1, k + 1):
            v = (u + d) % n
            edges.add(tuple(sorted((u, v))))
    while len(edges) < n * k + shortcuts:
        u, v = (int(x) for x in rng.integers(0, n, size=2))
        if u != v:
            edges.add(tuple(sorted((u, v))))
    rows, cols = [], []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])
    return coo_array((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()


def scale_free_graph(n: int = 360, m: int = 3, seed: int = 19) -> csr_array:
    """Small Barabasi--Albert-style preferential-attachment graph."""
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    degree = np.zeros(n, dtype=float)
    initial = m + 1
    for u in range(initial):
        for v in range(u + 1, initial):
            edges.add((u, v))
            degree[u] += 1
            degree[v] += 1
    for v in range(initial, n):
        probabilities = degree[:v] / degree[:v].sum()
        targets = rng.choice(v, size=m, replace=False, p=probabilities)
        for u in targets:
            edge = (int(u), v)
            edges.add(edge)
            degree[u] += 1
            degree[v] += 1
    rows, cols = [], []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])
    return coo_array((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()


def spy(ax: plt.Axes, matrix: csr_array, color: str = RED, size: float = 0.45) -> None:
    coo = matrix.tocoo()
    ax.scatter(coo.col, coo.row, s=size, c=color, marker="s", linewidths=0, rasterized=True)
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C7CDD2")
        spine.set_linewidth(0.7)


def add_stats(ax: plt.Axes, matrix: csr_array) -> None:
    density = 100.0 * matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    ax.text(
        0.5,
        -0.065,
        rf"$n={matrix.shape[0]:,}$   nnz$={matrix.nnz:,}$   density$={density:.2f}\%$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color=SOFT,
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_pattern_atlas() -> None:
    _, _, unstructured = unstructured_mesh()
    matrices = [
        ("Structured CFD", "5-point convection--diffusion", structured_cfd(35, 35), RED),
        ("Multiphysics CFD", "3 variables per cell", block_cfd(14, 14), ORANGE),
        ("Unstructured CFD", "mesh-neighbor coupling", unstructured, RED),
        ("Road / grid graph", "local planar connections", grid_graph(35, 35), ORANGE),
        ("Small-world graph", "bands plus shortcuts", small_world_graph(), RED),
        ("Scale-free graph", "hub-dominated degrees", scale_free_graph(), ORANGE),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 6.9), constrained_layout=True)
    for ax, (title, subtitle, matrix, color) in zip(axes.ravel(), matrices):
        spy(ax, matrix, color=color, size=0.55)
        ax.set_title(title, fontsize=12, color=INK, fontweight="bold", y=1.075, pad=0)
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=SOFT)
        add_stats(ax, matrix)
    save(fig, "sparse-pattern-atlas")


def make_mesh_mapping() -> None:
    points, tri, matrix = unstructured_mesh(n_points=170)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.7, 4.45), gridspec_kw={"width_ratios": [1.05, 1]})
    left.triplot(points[:, 0], points[:, 1], tri.simplices, color="#AAB3BA", linewidth=0.45)
    left.scatter(points[:, 0], points[:, 1], s=8, c=RED, edgecolors="white", linewidths=0.25, zorder=3)
    left.set_aspect("equal")
    left.set_xticks([])
    left.set_yticks([])
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_title("Unstructured CFD mesh", color=INK, fontsize=13, fontweight="bold")
    left.text(0.5, -0.02, "A basis function couples only to neighboring elements", transform=left.transAxes, ha="center", va="top", color=SOFT, fontsize=8.5)

    spy(right, matrix, color=RED, size=1.0)
    right.set_title("Finite-element sparsity pattern", color=INK, fontsize=13, fontweight="bold")
    right.text(0.5, -0.02, "Geometry becomes an irregular but still local row structure", transform=right.transAxes, ha="center", va="top", color=SOFT, fontsize=8.5)

    fig.subplots_adjust(wspace=0.19, bottom=0.09)
    save(fig, "cfd-mesh-to-matrix")


def hub_layout(matrix: csr_array) -> np.ndarray:
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    order = np.argsort(-degree)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    theta = (rank * 2.399963229728653) % (2 * np.pi)
    radius = 0.08 + 0.92 * np.sqrt(rank / max(1, len(order) - 1))
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def make_graph_mapping() -> None:
    matrix = scale_free_graph(n=190, m=3)
    degree = np.diff(matrix.indptr)
    order = np.argsort(-degree)
    ordered = matrix[order][:, order]
    pos = hub_layout(matrix)
    coo = matrix.tocoo()
    segments = [(pos[u], pos[v]) for u, v in zip(coo.row, coo.col) if u < v]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.7, 4.45), gridspec_kw={"width_ratios": [1.08, 1]})
    left.add_collection(LineCollection(segments, colors="#B7BFC5", linewidths=0.35, alpha=0.50))
    sizes = 7.0 + 1.5 * degree
    left.scatter(pos[:, 0], pos[:, 1], s=sizes, c=np.log1p(degree), cmap="Oranges", edgecolors="white", linewidths=0.25, zorder=3)
    left.set_xlim(-1.07, 1.07)
    left.set_ylim(-1.07, 1.07)
    left.set_aspect("equal")
    left.set_xticks([])
    left.set_yticks([])
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_title("Scale-free graph", color=INK, fontsize=13, fontweight="bold")
    left.text(0.5, -0.02, "A few hubs own much more work than most vertices", transform=left.transAxes, ha="center", va="top", color=SOFT, fontsize=8.5)

    spy(right, ordered, color=ORANGE, size=1.05)
    right.set_title("Adjacency matrix, degree ordered", color=INK, fontsize=13, fontweight="bold")
    right.add_patch(Rectangle((-0.5, -0.5), 25, ordered.shape[0], fill=False, edgecolor=RED, linewidth=1.4))
    right.text(0.5, -0.02, "Dense-looking hub rows create severe row-length imbalance", transform=right.transAxes, ha="center", va="top", color=SOFT, fontsize=8.5)

    fig.subplots_adjust(wspace=0.18, bottom=0.09)
    save(fig, "graph-to-matrix")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )
    make_pattern_atlas()
    make_mesh_mapping()
    make_graph_mapping()
    print(f"Wrote sparse-matrix figures to {OUT}")


if __name__ == "__main__":
    main()
