import numpy as np
import pytest

import spcraft


def test_coo_to_csr():
    # The same 2 x 3 sparse matrix in COO format:
    #
    #   [0  2  0]
    #   [4  0  3]
    coo = spcraft.coo_matrix(
        row_indices=[0, 1, 1],
        column_indices=[1, 0, 2],
        values=np.array([2, 4, 3], dtype=np.float32),
        shape=(2, 3),
    )

    assert coo.shape == (2, 3)
    assert coo.nnz == 3

    csr = coo.to_csr()

    assert csr.row_offsets.tolist() == [0, 1, 3]
    assert csr.column_indices.tolist() == [1, 0, 2]
    assert csr.values.tolist() == [2, 4, 3]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_coo_entry_storage_copies_at_both_boundaries(dtype):
    rows = np.array([1, 0, 1, 1], dtype=np.int32)
    columns = np.array([2, 0, 2, 1], dtype=np.int32)
    values = np.array([3, 4, -3, 0], dtype=dtype)
    expected_rows, expected_columns, expected_values = rows.copy(), columns.copy(), values.copy()
    matrix = spcraft.coo_matrix(rows, columns, values, shape=(2, 3))
    rows[:] = 0
    columns[:] = 0
    values[:] = 99
    np.testing.assert_array_equal(matrix.row_indices, expected_rows)
    np.testing.assert_array_equal(matrix.column_indices, expected_columns)
    np.testing.assert_array_equal(matrix.values, expected_values)
    assert matrix.values.dtype == dtype
    for component in (matrix.row_indices, matrix.column_indices, matrix.values):
        component[:] = 99
    np.testing.assert_array_equal(matrix.row_indices, expected_rows)
    np.testing.assert_array_equal(matrix.column_indices, expected_columns)
    np.testing.assert_array_equal(matrix.values, expected_values)
    clone = matrix.clone()
    del matrix
    np.testing.assert_array_equal(clone.values, expected_values)
    # Conversion retains duplicate coordinates and explicit zero entries.
    csr = clone.to_csr()
    assert csr.nnz == 4
    np.testing.assert_array_equal(csr.row_offsets, [0, 1, 4])
    np.testing.assert_array_equal(csr.column_indices, [0, 2, 2, 1])
    np.testing.assert_array_equal(csr.values, np.array([4, 3, -3, 0], dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_coo_entry_storage(dtype):
    matrix = spcraft.coo_matrix([], [], np.array([], dtype=dtype), shape=(2, 3))
    assert matrix.shape == (2, 3)
    assert matrix.nnz == 0
    assert matrix.row_indices.size == matrix.column_indices.size == matrix.values.size == 0
    assert matrix.clone().shape == (2, 3)
    np.testing.assert_array_equal(matrix.to_csr().row_offsets, [0, 0, 0])
