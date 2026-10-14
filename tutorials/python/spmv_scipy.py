#!/usr/bin/env python3
"""Small, inspectable SciPy examples for arithmetic and graph SpMV."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array


def arithmetic_spmv() -> None:
    a = csr_array(
        np.array(
            [
                [10.0, 0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [4.0, 0.0, 5.0, 6.0],
                [0.0, 0.0, 0.0, 7.0],
            ]
        )
    )
    x = np.array([1.0, 2.0, -1.0, 0.5])
    y = a @ x

    print("Arithmetic SpMV")
    print("indptr :", a.indptr)
    print("indices:", a.indices)
    print("data   :", a.data)
    print("x      :", x)
    print("A @ x  :", y)
    np.testing.assert_allclose(y, a.toarray() @ x)


def bfs_with_sparse_frontiers() -> None:
    # A[v, u] = 1 means an edge u -> v.  Multiplying A @ frontier
    # therefore gathers outgoing neighbors of the current frontier.
    sources = np.array([0, 0, 1, 2, 2, 3])
    destinations = np.array([1, 2, 3, 3, 4, 5])
    adjacency = csr_array(
        (np.ones_like(sources, dtype=np.int8), (destinations, sources)),
        shape=(6, 6),
    )

    frontier = np.zeros(6, dtype=bool)
    frontier[0] = True
    visited = frontier.copy()
    level = 0

    print("\nBFS via masked frontier SpMV")
    while frontier.any():
        print(f"level {level}: {np.flatnonzero(frontier).tolist()}")
        # Integer plus-times followed by >0 is equivalent to Boolean OR-AND
        # for reachability when the inputs are structural 0/1 values.
        candidates = (adjacency @ frontier.astype(np.int8)) > 0
        frontier = candidates & ~visited
        visited |= frontier
        level += 1


if __name__ == "__main__":
    arithmetic_spmv()
    bfs_with_sparse_frontiers()
