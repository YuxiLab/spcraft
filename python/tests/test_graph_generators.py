import numpy as np
import pytest
import scipy.sparse as sp

import spcraft


def to_scipy(matrix):
    return sp.csr_array(
        (matrix.values, matrix.column_indices, matrix.row_offsets),
        shape=matrix.shape,
    )


def assert_simple_undirected_graph(matrix, vertices, dtype):
    assert matrix.shape == (vertices, vertices)
    assert matrix.values.dtype == dtype

    scipy_matrix = to_scipy(matrix)
    assert scipy_matrix.diagonal().sum() == 0
    assert (scipy_matrix != scipy_matrix.T).nnz == 0
    np.testing.assert_array_equal(matrix.values, np.ones(matrix.nnz, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gen_er_graph(dtype):
    graph = spcraft.gen_er_graph(64, 8.0, 42, dtype=dtype)
    repeated = spcraft.gen_er_graph(64, 8.0, 42, dtype=dtype)

    assert_simple_undirected_graph(graph, 64, dtype)
    np.testing.assert_array_equal(graph.row_offsets, repeated.row_offsets)
    np.testing.assert_array_equal(graph.column_indices, repeated.column_indices)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gen_rmat(dtype):
    graph = spcraft.gen_rmat(6, 8, 42, dtype=dtype)
    repeated = spcraft.gen_rmat(6, 8, 42, dtype=dtype)

    assert_simple_undirected_graph(graph, 1 << 6, dtype)
    np.testing.assert_array_equal(graph.row_offsets, repeated.row_offsets)
    np.testing.assert_array_equal(graph.column_indices, repeated.column_indices)


@pytest.mark.parametrize("generator", [spcraft.gen_er_graph, spcraft.gen_rmat])
def test_graph_generator_dtype_validation(generator):
    with pytest.raises(TypeError, match="dtype"):
        generator(16 if generator is spcraft.gen_er_graph else 4, 4, 42, dtype=np.int32)
