import numpy as np

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
