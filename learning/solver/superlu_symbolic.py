"""Structural stages of an educational SuperLU-style sparse direct solver.

Source guide (relative to workspace):
  superlu/SRC/get_perm_c.c   -> column ordering
  superlu/SRC/sp_preorder.c  -> etree and postordering
  superlu/SRC/relax_snode.c  -> relaxed supernode planning

We implement exact minimum degree on the column-intersection graph of A^T*A,
not COLAMD. A^T*A is NEVER formed numerically. Its graph can still be much
denser than A; this intentionally transparent algorithm targets small examples.
The symbolic patterns describe Cholesky of the structural A^T*A envelope.
They guide scheduling, not fixed LU allocation: numerical row pivoting can
change actual LU structure. Numeric storage therefore grows dynamically.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from time import perf_counter
from typing import Sequence

if __package__:
    from .sparse_solver import CSR
else:
    from sparse_solver import CSR


def canonicalize(A: CSR) -> CSR:
    """Sum duplicate entries, sort column indices, and remove numerical zeros."""
    if A.shape[0] != A.shape[1]:
        raise ValueError("The direct solver requires a square matrix")
    return CSR.from_coo(A.shape, (
        (i, A.indices[k], A.data[k])
        for i in range(A.shape[0])
        for k in range(A.indptr[i], A.indptr[i + 1])
    ))


@dataclass(frozen=True)
class CSC:
    """Compressed columns, the input layout used by SuperLU's factorization."""

    n: int
    data: tuple[float, ...]
    row_indices: tuple[int, ...]
    col_ptr: tuple[int, ...]

    @classmethod
    def from_csr(cls, A: CSR) -> CSC:
        n = A.shape[0]
        columns: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            for k in range(A.indptr[i], A.indptr[i + 1]):
                columns[A.indices[k]].append((i, A.data[k]))
        data, rows, ptr = [], [], [0]
        for column in columns:
            for i, value in column:
                rows.append(i)
                data.append(value)
            ptr.append(len(data))
        return cls(n, tuple(data), tuple(rows), tuple(ptr))

    @property
    def pattern(self) -> tuple:
        return self.n, self.col_ptr, self.row_indices

    def norm_1(self) -> float:
        return max((fsum(abs(v) for v in self.data[self.col_ptr[j]:self.col_ptr[j + 1]])
                    for j in range(self.n)), default=0.0)


def column_graph(A: CSR) -> list[set[int]]:
    """Columns sharing a nonzero row are adjacent in structural A^T*A."""
    graph = [set() for _ in range(A.shape[1])]
    for i in range(A.shape[0]):
        columns = set(A.indices[A.indptr[i]:A.indptr[i + 1]])
        for j in columns:
            graph[j].update(columns - {j})
    return graph


def minimum_degree(graph: Sequence[set[int]]) -> list[int]:
    """Exact greedy minimum degree, with fill edges and deterministic ties."""
    work = [neighbors.copy() for neighbors in graph]
    active = set(range(len(graph)))
    order = []
    while active:
        j = min(active, key=lambda k: (len(work[k]), k))
        neighbors = work[j].copy()
        for i in neighbors:
            work[i].discard(j)
            work[i].update(neighbors - {i})
        active.remove(j)
        work[j].clear()
        order.append(j)
    return order


def permute_graph(graph: Sequence[set[int]], order: Sequence[int]) -> list[set[int]]:
    inverse = [0] * len(order)
    for new, old in enumerate(order):
        inverse[old] = new
    return [{inverse[i] for i in graph[old]} for old in order]


def symbolic_cholesky(graph: Sequence[set[int]]) -> tuple[list[int], list[set[int]]]:
    """Simulate structural elimination, returning etree and lower-column sets.

    etree[j] is the first higher neighbor after fill; n denotes a root.
    The set for column j includes its diagonal j.
    """
    n = len(graph)
    work = [neighbors.copy() for neighbors in graph]
    etree, columns = [n] * n, []
    for j in range(n):
        higher = {i for i in work[j] if i > j}
        columns.append(higher | {j})
        if higher:
            etree[j] = min(higher)
        for i in higher:
            work[i].discard(j)
            work[i].update(higher - {i})
        work[j].clear()
    return etree, columns


def tree_postorder(etree: Sequence[int]) -> list[int]:
    """Iterative child-before-parent traversal; works for an elimination forest."""
    n = len(etree)
    children = [[] for _ in range(n + 1)]
    for j, parent in enumerate(etree):
        children[parent].append(j)
    result = []
    stack = [(root, False) for root in reversed(children[n])]
    while stack:
        node, visited = stack.pop()
        if visited:
            result.append(node)
        else:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(children[node]))
    return result


def strict_supernodes(columns: Sequence[set[int]], max_width: int) -> list[tuple[int, int]]:
    """Consecutive columns with identical below-block row structure can share storage."""
    n = len(columns)
    blocks, start = [], 0
    while start < n:
        end = start + 1
        while (end < n and end - start < max_width
               and columns[end - 1] - {end - 1} == columns[end]):
            end += 1
        blocks.append((start, end))
        start = end
    return blocks


def relaxed_supernodes(etree: Sequence[int], max_width: int) -> list[tuple[int, int]]:
    """Amalgamate maximal small complete subtrees of a postordered etree."""
    n = len(etree)
    sizes = [1] * n
    for j, parent in enumerate(etree):
        if parent < n:
            sizes[parent] += sizes[j]
    groups = []
    for j, parent in enumerate(etree):
        if sizes[j] <= max_width and (parent == n or sizes[parent] > max_width):
            # Complete subtrees occupy consecutive indices in a DFS postorder.
            groups.append((j + 1 - sizes[j], j + 1))
    return sorted(groups)


def panel_schedule(n: int, width: int, relaxed: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Use relaxed-subtree boundaries, packing other columns into bounded panels."""
    starts = {s: e for s, e in relaxed}
    panels, start = [], 0
    while start < n:
        if start in starts:
            end = min(starts[start], start + width)
            # Split relaxed groups that are wider than the requested panel width.
            if end < starts[start]:
                starts[end] = starts[start]
        else:
            next_group = min((s for s in starts if s > start), default=n)
            end = min(n, start + width, next_group)
        panels.append((start, end))
        start = end
    return panels


@dataclass(frozen=True)
class SymbolicAnalysis:
    pattern: tuple
    ordering: str
    # New-to-old convention: ordered column k is original column col_order[k].
    # This is the inverse convention of SuperLU's perm_c array.
    col_order: tuple[int, ...]
    etree: tuple[int, ...]
    column_patterns: tuple[frozenset[int], ...]
    strict_groups: tuple[tuple[int, int], ...]
    relaxed_groups: tuple[tuple[int, int], ...]
    panels: tuple[tuple[int, int], ...]
    panel_size: int
    relax: int
    postordered: bool
    ordering_seconds: float
    symbolic_seconds: float

    def check_pattern(self, A: CSR) -> None:
        if CSC.from_csr(A).pattern != self.pattern:
            raise ValueError("Symbolic reuse requires the same canonical sparsity pattern")

    @property
    def envelope_nnz(self) -> int:
        return sum(len(column) for column in self.column_patterns)


def analyze(
    A: CSR,
    *,
    ordering: str = "minimum_degree",
    col_order: Sequence[int] | None = None,
    postorder: bool = True,
    panel_size: int = 4,
    relax: int = 4,
) -> SymbolicAnalysis:
    """Build a value-independent plan, reusable when only matrix values change."""
    if (not isinstance(panel_size, int) or not isinstance(relax, int)
            or panel_size < 1 or relax < 1):
        raise ValueError("panel_size and relax must be positive integers")
    A = canonicalize(A)
    n = A.shape[0]
    start = perf_counter()
    graph = column_graph(A)
    if col_order is not None:
        order = list(col_order)
        if sorted(order) != list(range(n)):
            raise ValueError("col_order must be a permutation of 0..n-1")
        ordering = "user"
    elif ordering == "natural":
        order = list(range(n))
    elif ordering == "minimum_degree":
        order = minimum_degree(graph)
    else:
        raise ValueError("ordering must be 'natural' or 'minimum_degree'")
    ordering_seconds = perf_counter() - start

    start = perf_counter()
    etree, columns = symbolic_cholesky(permute_graph(graph, order))
    if postorder:
        post = tree_postorder(etree)
        order = [order[j] for j in post]
        etree, columns = symbolic_cholesky(permute_graph(graph, order))
    strict = strict_supernodes(columns, panel_size)
    relaxed = relaxed_supernodes(etree, relax) if postorder else []
    panels = panel_schedule(n, panel_size, relaxed)
    return SymbolicAnalysis(
        CSC.from_csr(A).pattern, ordering, tuple(order), tuple(etree),
        tuple(frozenset(column) for column in columns), tuple(strict),
        tuple(relaxed), tuple(panels), panel_size, relax, postorder,
        ordering_seconds, perf_counter() - start,
    )
