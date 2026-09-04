"""Python interface for the SPCraft C++ library.

This first interface intentionally copies arrays at the Python/C++ boundary.
The standalone C++ library does not depend on this package or on Python.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Optional, Union

import numpy as np

from ._spcraft import (
    CooMatrixF32,
    CooMatrixF64,
    CsrMatrixF32,
    CsrMatrixF64,
    _gen_er_graph_f32,
    _gen_er_graph_f64,
    _gen_rmat_f32,
    _gen_rmat_f64,
)


def _shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("shape must contain exactly two dimensions")
    rows, columns = (int(dimension) for dimension in shape)
    if rows < 0 or columns < 0:
        raise ValueError("matrix dimensions must be non-negative")
    return rows, columns


def _values(values: Any, dtype: Optional[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = array.astype(np.float64)
    return np.ascontiguousarray(array)


def coo_matrix(
    row_indices: Any,
    column_indices: Any,
    values: Any,
    shape: tuple[int, int],
    *,
    dtype: Optional[Any] = None,
) -> Union[CooMatrixF32, CooMatrixF64]:
    """Create an owning SPCraft COO matrix by copying the supplied arrays."""

    rows, columns = _shape(shape)
    row_array = np.ascontiguousarray(row_indices, dtype=np.int32)
    column_array = np.ascontiguousarray(column_indices, dtype=np.int32)
    value_array = _values(values, dtype)

    matrix_type = CooMatrixF32 if value_array.dtype == np.float32 else CooMatrixF64
    return matrix_type.from_arrays(row_array, column_array, value_array, rows, columns)


def csr_matrix(
    row_offsets: Any,
    column_indices: Any,
    values: Any,
    shape: tuple[int, int],
    *,
    dtype: Optional[Any] = None,
) -> Union[CsrMatrixF32, CsrMatrixF64]:
    """Create an owning SPCraft CSR matrix by copying the supplied arrays."""

    rows, columns = _shape(shape)
    offset_array = np.ascontiguousarray(row_offsets, dtype=np.int32)
    column_array = np.ascontiguousarray(column_indices, dtype=np.int32)
    value_array = _values(values, dtype)

    matrix_type = CsrMatrixF32 if value_array.dtype == np.float32 else CsrMatrixF64
    return matrix_type.from_arrays(offset_array, column_array, value_array, rows, columns)


def gen_er_graph(
    vertices: int,
    expected_degree: float,
    seed: int,
    *,
    dtype: Any = np.float64,
) -> Union[CsrMatrixF32, CsrMatrixF64]:
    """Generate an undirected Erdos-Renyi graph directly in CSR format."""

    normalized_dtype = np.dtype(dtype)
    if normalized_dtype == np.float32:
        return _gen_er_graph_f32(vertices, expected_degree, seed)
    if normalized_dtype == np.float64:
        return _gen_er_graph_f64(vertices, expected_degree, seed)
    raise TypeError("dtype must be numpy.float32 or numpy.float64")


def gen_rmat(
    scale: int,
    edge_factor: int,
    seed: int,
    *,
    a: float = 0.57,
    b: float = 0.19,
    c: float = 0.19,
    d: float = 0.05,
    dtype: Any = np.float64,
) -> Union[CsrMatrixF32, CsrMatrixF64]:
    """Generate a symmetric, deduplicated R-MAT graph in CSR format."""

    normalized_dtype = np.dtype(dtype)
    arguments = (scale, edge_factor, seed, a, b, c, d)
    if normalized_dtype == np.float32:
        return _gen_rmat_f32(*arguments)
    if normalized_dtype == np.float64:
        return _gen_rmat_f64(*arguments)
    raise TypeError("dtype must be numpy.float32 or numpy.float64")


def from_matrix_market(
    filename: Union[str, PathLike[str]], *, dtype: Any = np.float64
) -> Union[CooMatrixF32, CooMatrixF64]:
    """Read a Matrix Market file into an owning COO matrix."""

    normalized_dtype = np.dtype(dtype)
    if normalized_dtype == np.float32:
        return CooMatrixF32.from_matrix_market(str(filename))
    if normalized_dtype == np.float64:
        return CooMatrixF64.from_matrix_market(str(filename))
    raise TypeError("dtype must be numpy.float32 or numpy.float64")


def from_scipy(
    matrix: Any,
    *,
    dtype: Optional[Any] = None,
) -> Union[CsrMatrixF32, CsrMatrixF64, CooMatrixF32, CooMatrixF64]:
    """Convert a SciPy sparse matrix/array (CSR or COO) to an SPCraft matrix."""
    import scipy.sparse as sp

    target_dtype = dtype or getattr(matrix, "dtype", None)
    if target_dtype is not None and np.dtype(target_dtype) not in (np.float32, np.float64):
        target_dtype = np.float64

    if isinstance(matrix, (sp.csr_matrix, sp.csr_array)):
        return csr_matrix(
            matrix.indptr,
            matrix.indices,
            matrix.data,
            shape=matrix.shape,
            dtype=target_dtype,
        )
    if isinstance(matrix, (sp.coo_matrix, sp.coo_array)):
        return coo_matrix(
            matrix.row,
            matrix.col,
            matrix.data,
            shape=matrix.shape,
            dtype=target_dtype,
        )
    if hasattr(matrix, "tocsr"):
        csr = matrix.tocsr()
        return csr_matrix(
            csr.indptr,
            csr.indices,
            csr.data,
            shape=csr.shape,
            dtype=target_dtype or csr.dtype,
        )
    raise TypeError(f"Unsupported matrix type: {type(matrix)}")


def spmv(matrix: Any, x: Any) -> np.ndarray:
    """Multiply a sparse matrix by dense vector x (y = A @ x) using SPCraft."""
    x_array = np.ascontiguousarray(x)
    if isinstance(matrix, (CsrMatrixF32, CsrMatrixF64)):
        expected_dtype = matrix.values.dtype
        if x_array.dtype != expected_dtype:
            x_array = x_array.astype(expected_dtype)
        return matrix.spmv(x_array)
    if isinstance(matrix, (CooMatrixF32, CooMatrixF64)):
        csr = matrix.to_csr()
        expected_dtype = csr.values.dtype
        if x_array.dtype != expected_dtype:
            x_array = x_array.astype(expected_dtype)
        return csr.spmv(x_array)
    raise TypeError(f"Unsupported matrix type for SPCraft SpMV: {type(matrix)}")


__all__ = [
    "CooMatrixF32",
    "CooMatrixF64",
    "CsrMatrixF32",
    "CsrMatrixF64",
    "coo_matrix",
    "csr_matrix",
    "gen_er_graph",
    "gen_rmat",
    "from_matrix_market",
    "from_scipy",
    "spmv",
]
