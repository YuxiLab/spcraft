"""The algorithms layered on SpMV, through the Python boundary.

Each check has an independent reference: NumPy/SciPy, or a closed form.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import spcraft


def _csr(dense, dtype=np.float64):
    matrix = sp.csr_array(np.asarray(dense, dtype=dtype))
    return spcraft.csr_matrix(
        matrix.indptr, matrix.indices, matrix.data, shape=matrix.shape, dtype=dtype
    )


def _adjacency(vertices, edges):
    dense = np.zeros((vertices, vertices))
    for source, target in edges:
        dense[source, target] = 1.0
    return dense


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spgemm_matches_scipy(dtype):
    left = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    right = np.array([[4.0, 0.0], [0.0, 5.0], [6.0, 7.0]])

    product = _csr(left, dtype).spgemm(_csr(right, dtype))
    expected = sp.csr_array(left) @ sp.csr_array(right)

    assert product.shape == (2, 2)
    dense = np.zeros((2, 2))
    offsets, columns, values = (
        product.row_offsets,
        product.column_indices,
        product.values,
    )
    for row in range(2):
        for position in range(offsets[row], offsets[row + 1]):
            dense[row, columns[position]] = values[position]
    np.testing.assert_allclose(dense, expected.toarray(), rtol=1e-5)


def test_pagerank_ranks_sum_to_one_and_match_the_eigenvector():
    edges = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 0), (3, 1), (3, 4), (4, 5), (5, 4), (5, 0)]
    vertices = 6
    damping = 0.85
    adjacency = _adjacency(vertices, edges)

    ranks, info = _csr(adjacency).pagerank(
        damping=damping, tolerance=1e-14, max_iterations=500
    )

    assert info["converged"]
    assert np.isclose(ranks.sum(), 1.0)

    # Vertex 3 has no in-links, so its rank is exactly the teleport floor.
    assert np.isclose(ranks[3], (1.0 - damping) / vertices)

    # Independent reference: the dominant eigenvector of the dense Google matrix.
    out_degree = adjacency.sum(axis=1)
    transition = np.zeros_like(adjacency)
    for source in range(vertices):
        if out_degree[source] > 0:
            transition[:, source] = adjacency[source, :] / out_degree[source]
    google = damping * transition + (1.0 - damping) / vertices
    values, vectors = np.linalg.eig(google)
    dominant = vectors[:, np.argmax(values.real)].real
    dominant = dominant / dominant.sum()
    np.testing.assert_allclose(ranks, dominant, atol=1e-10)


def test_pagerank_handles_a_dangling_vertex():
    # Vertex 2 has no out-links; its rank must be redistributed, not lost.
    ranks, info = _csr(_adjacency(3, [(0, 2), (1, 2)])).pagerank(tolerance=1e-14)
    assert info["converged"]
    assert np.isclose(ranks.sum(), 1.0)
    assert ranks[2] > ranks[0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cg_solves_a_small_spd_system(dtype):
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0], dtype=dtype)

    tolerance = 1e-5 if dtype == np.float32 else 1e-10
    x, info = _csr(matrix, dtype).solve_cg(rhs, tolerance=tolerance)

    assert info["converged"]
    assert not info["breakdown"]
    assert info["iterations"] <= 2  # CG terminates in at most n steps
    np.testing.assert_allclose(x, np.linalg.solve(matrix, rhs), rtol=1e-4)


def test_cg_matches_numpy_on_a_laplacian():
    grid = 8
    size = grid * grid
    dense = np.zeros((size, size))
    for row in range(grid):
        for column in range(grid):
            index = row * grid + column
            dense[index, index] = 4.0
            for neighbour_row, neighbour_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if 0 <= neighbour_row < grid and 0 <= neighbour_column < grid:
                    dense[index, neighbour_row * grid + neighbour_column] = -1.0

    rhs = np.sin(0.7 * np.arange(size)) + 0.5
    x, info = _csr(dense).solve_cg(rhs, tolerance=1e-12, max_iterations=500)

    assert info["converged"]
    np.testing.assert_allclose(x, np.linalg.solve(dense, rhs), atol=1e-8)


def test_cg_reports_a_breakdown_on_an_indefinite_matrix():
    _, info = _csr(np.diag([1.0, -1.0])).solve_cg(np.array([1.0, 1.0]))
    assert info["breakdown"]
    assert not info["converged"]


def test_cg_rejects_a_mismatched_right_hand_side():
    with pytest.raises(Exception):
        _csr(np.eye(2)).solve_cg(np.array([1.0, 2.0, 3.0]))


def test_power_iteration_finds_the_dominant_eigenpair():
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    eigenvalue, eigenvector, info = _csr(matrix).power_iteration()

    assert info["converged"]
    assert np.isclose(eigenvalue, 3.0)
    assert np.isclose(np.linalg.norm(eigenvector), 1.0)
    # The eigenvector is only defined up to sign.
    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    assert min(
        np.abs(eigenvector - expected).max(), np.abs(eigenvector + expected).max()
    ) < 1e-8


def test_power_iteration_matches_numpy_and_handles_a_negative_eigenvalue():
    matrix = np.diag([-5.0, 1.0, 2.0])
    eigenvalue, _, info = _csr(matrix).power_iteration()

    assert info["converged"]
    expected = max(np.linalg.eigvals(matrix), key=abs).real
    assert np.isclose(eigenvalue, expected)
    assert np.isclose(eigenvalue, -5.0)


def test_power_iteration_does_not_claim_convergence_without_a_gap():
    # +/-1 have equal magnitude, so nothing decays and the cap must stop it.
    _, _, info = _csr(np.diag([1.0, -1.0])).power_iteration(max_iterations=50)
    assert not info["converged"]
    assert info["iterations"] == 50


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spgemm_canonicalizes_csr_without_mutating_inputs(dtype):
    # Unsorted columns, duplicate coordinates, cancellation and stored zeros.
    left = sp.csr_array((np.array([2, 1, -2, 3, 0], dtype=dtype),
                         np.array([2, 0, 2, 1, 0]), np.array([0, 3, 5])), shape=(2, 3))
    right = sp.csr_array((np.array([4, 6, 5, 7], dtype=dtype),
                          np.array([1, 0, 1, 0]), np.array([0, 1, 2, 4])), shape=(3, 2))
    a = spcraft.csr_matrix(left.indptr, left.indices, left.data, shape=left.shape)
    b = spcraft.csr_matrix(right.indptr, right.indices, right.data, shape=right.shape)
    result = a.spgemm(b)
    actual = sp.csr_array((result.values, result.column_indices, result.row_offsets),
                          shape=result.shape)
    np.testing.assert_allclose(actual.toarray(), (left @ right).toarray(), rtol=1e-5)
    np.testing.assert_array_equal(a.column_indices, left.indices)
    np.testing.assert_array_equal(a.values, left.data)
    np.testing.assert_array_equal(b.column_indices, right.indices)
    assert result.values.dtype == dtype
    assert result.nnz == 4  # Both rows reach both columns, including zero-valued paths.


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spgemm_dcsc_boundary_empty_and_mismatched_shapes(dtype):
    a = _csr(np.zeros((2, 0)), dtype)
    b = _csr(np.zeros((0, 3)), dtype)
    result = a.spgemm(b)
    assert result.shape == (2, 3) and result.nnz == 0
    np.testing.assert_array_equal(result.row_offsets, [0, 0, 0])
    with pytest.raises(ValueError, match="dimensions"):
        a.spgemm(_csr(np.zeros((1, 3)), dtype))
