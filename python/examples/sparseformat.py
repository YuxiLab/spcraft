#!/usr/bin/env python3
"""Compare and test SpMV (Sparse Matrix-Vector Multiplication) between SciPy and SPCraft."""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import spcraft as spc


def test_small_spmv() -> None:
    print("=" * 60)
    print("1. Small Inspectable SpMV Test")
    print("=" * 60)

    # 4 x 4 sparse matrix
    dense = np.array(
        [
            [10.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 0.0, 7.0],
        ],
        dtype=np.float64,
    )
    x = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float64)

    # SciPy CSR
    scipy_csr = sp.csr_matrix(dense)
    y_scipy = scipy_csr @ x

    # SPCraft CSR (converted from scipy or created directly)
    spc_csr = spc.from_scipy(scipy_csr)
    y_spc_matmul = spc_csr @ x
    y_spc_method = spc_csr.spmv(x)
    y_spc_top = spc.spmv(spc_csr, x)

    print(f"Matrix shape: {spc_csr.shape}, nnz: {spc_csr.nnz}")
    print("x             :", x)
    print("SciPy   A @ x :", y_scipy)
    print("SPCraft    A @ x :", y_spc_matmul)
    print("SPCraft   spmv(x):", y_spc_method)

    # Assert correctness
    np.testing.assert_allclose(y_spc_matmul, y_scipy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_spc_method, y_scipy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_spc_top, y_scipy, rtol=1e-12, atol=1e-12)
    print("✓ Small SpMV test passed!\n")


def test_random_spmv_benchmark(
    rows: int = 20_000,
    cols: int = 20_000,
    density: float = 0.001,
    dtype: type = np.float64,
    iterations: int = 50,
    warmup: int = 5,
) -> None:
    dtype_name = np.dtype(dtype).name
    print("=" * 60)
    print(f"2. Random Matrix SpMV Benchmark ({rows}x{cols}, density={density:.2%}, dtype={dtype_name})")
    print("=" * 60)

    # Generate random CSR matrix
    print("Generating random sparse matrix...")
    scipy_csr = sp.random(rows, cols, density=density, format="csr", dtype=dtype, random_state=42)
    nnz = scipy_csr.nnz
    print(f"Nonzeros (nnz): {nnz:,} (avg {nnz/rows:.1f} per row)")

    # Input vector
    rng = np.random.default_rng(123)
    x = rng.standard_normal(cols).astype(dtype)

    # Convert to SPCraft CSR
    spc_csr = spc.from_scipy(scipy_csr)

    # --- Correctness Check ---
    y_scipy = scipy_csr @ x
    y_spc = spc_csr @ x
    max_err = np.max(np.abs(y_scipy - y_spc))
    rel_err = max_err / (np.max(np.abs(y_scipy)) + 1e-14)
    print(f"Max absolute error: {max_err:.2e}, Relative error: {rel_err:.2e}")
    np.testing.assert_allclose(y_spc, y_scipy, rtol=1e-5, atol=1e-6)
    print("✓ Correctness verification passed!")

    # --- Warmup ---
    for _ in range(warmup):
        _ = scipy_csr @ x
        _ = spc_csr @ x

    # --- SciPy Timing ---
    start = time.perf_counter()
    for _ in range(iterations):
        _ = scipy_csr @ x
    scipy_ms = (time.perf_counter() - start) / iterations * 1000.0

    # --- SPCraft Timing ---
    start = time.perf_counter()
    for _ in range(iterations):
        _ = spc_csr @ x
    spc_ms = (time.perf_counter() - start) / iterations * 1000.0

    # Performance metrics (2 * nnz FLOPs per SpMV)
    flops = 2.0 * nnz
    scipy_gflops = (flops / (scipy_ms * 1e-3)) / 1e9
    spc_gflops = (flops / (spc_ms * 1e-3)) / 1e9
    speedup = scipy_ms / spc_ms

    print(f"\nResults ({iterations} iterations averaged):")
    print(f"  SciPy SpMV    : {scipy_ms:8.3f} ms | {scipy_gflops:6.2f} GFLOPS")
    print(f"  SPCraft SpMV : {spc_ms:8.3f} ms | {spc_gflops:6.2f} GFLOPS")
    print(f"  Speedup       : {speedup:8.2f}x {'(faster)' if speedup > 1.0 else '(slower)'}\n")


if __name__ == "__main__":
    test_small_spmv()
    # Test single-precision (float32)
    test_random_spmv_benchmark(rows=10_000, cols=10_000, density=0.002, dtype=np.float32)
    # Test double-precision (float64)
    test_random_spmv_benchmark(rows=20_000, cols=20_000, density=0.001, dtype=np.float64)
