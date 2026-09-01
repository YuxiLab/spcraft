import numpy as np
import pytest
import scipy.sparse as sp

import spcraft


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_csr_spmv_small(dtype):
    # Matrix:
    # [ 2.0  0.0 -1.0  0.0  0.5 ]
    # [ 0.0  0.0  0.0  0.0  0.0 ]
    # [ 0.0  3.0  0.0 -2.0  0.0 ]
    # [-4.0  0.0  0.0  0.0  1.0 ]
    row_offsets = [0, 3, 3, 5, 7]
    col_indices = [0, 2, 4, 1, 3, 0, 4]
    values = np.array([2.0, -1.0, 0.5, 3.0, -2.0, -4.0, 1.0], dtype=dtype)
    shape = (4, 5)

    csr = spcraft.csr_matrix(row_offsets, col_indices, values, shape=shape, dtype=dtype)
    x = np.array([1.0, 2.0, -3.0, 4.0, 5.0], dtype=dtype)

    expected = np.array([7.5, 0.0, -2.0, 1.0], dtype=dtype)

    # Test .spmv()
    y1 = csr.spmv(x)
    np.testing.assert_allclose(y1, expected, rtol=1e-5, atol=1e-6)

    # Test @ operator (__matmul__)
    y2 = csr @ x
    np.testing.assert_allclose(y2, expected, rtol=1e-5, atol=1e-6)

    # Test top-level spmv
    y3 = spcraft.spmv(csr, x)
    np.testing.assert_allclose(y3, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spmv_against_scipy_random(dtype):
    rng = np.random.default_rng(42)
    m, n = 50, 40
    density = 0.15

    scipy_csr = sp.random(m, n, density=density, format="csr", dtype=dtype, random_state=42)
    spc_csr = spcraft.from_scipy(scipy_csr)

    x = rng.standard_normal(n).astype(dtype)

    expected = scipy_csr @ x
    actual_method = spc_csr.spmv(x)
    actual_matmul = spc_csr @ x
    actual_top = spcraft.spmv(spc_csr, x)

    np.testing.assert_allclose(actual_method, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_matmul, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_top, expected, rtol=1e-5, atol=1e-6)


def test_coo_spmv_via_conversion():
    row = [0, 1, 1]
    col = [1, 0, 2]
    data = [2.0, 4.0, 3.0]
    coo = spcraft.coo_matrix(row, col, data, shape=(2, 3), dtype=np.float64)
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    # y = [2*20, 4*10 + 3*30] = [40, 130]
    y = spcraft.spmv(coo, x)
    np.testing.assert_allclose(y, [40.0, 130.0])


def test_dimension_mismatch():
    csr = spcraft.csr_matrix([0, 1], [0], [1.0], shape=(1, 3), dtype=np.float64)
    with pytest.raises(Exception):
        csr.spmv(np.array([1.0, 2.0]))  # Expects len 3
