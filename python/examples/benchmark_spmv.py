#!/usr/bin/env python3
"""Comprehensive SpMV testing and benchmarking suite: SciPy vs SPCraft.

Features:
- Validates SpMV numerical correctness against SciPy reference.
- Measures execution times, GFLOPS throughput, and speedups.
- Tests across multiple matrix sizes, non-zero densities, and dtypes (float32, float64).
- Supports loading external Matrix Market (.mtx) files.
- Optional CSV export and visualization plots.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp

import spcraft as spc


def run_benchmark(
    matrix_scipy: sp.csr_matrix | sp.csr_array,
    dtype: type = np.float64,
    iterations: int = 50,
    warmup: int = 10,
    check_accuracy: bool = True,
) -> dict[str, float | str | int | bool]:
    """Benchmark a single CSR matrix with SciPy and SPCraft."""
    m, n = matrix_scipy.shape
    nnz = matrix_scipy.nnz

    # Generate random dense input vector
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n).astype(dtype)

    # Convert to SPCraft CSR
    matrix_spc = spc.from_scipy(matrix_scipy, dtype=dtype)

    # 1. Correctness Verification
    max_err = 0.0
    rel_err = 0.0
    passed = True
    if check_accuracy:
        y_scipy = matrix_scipy @ x
        y_spc = matrix_spc @ x
        max_err = float(np.max(np.abs(y_scipy - y_spc)))
        norm_scipy = float(np.max(np.abs(y_scipy))) + 1e-14
        rel_err = max_err / norm_scipy
        tol = 1e-4 if dtype == np.float32 else 1e-7
        passed = bool(rel_err < tol)

    # 2. Warmup
    for _ in range(warmup):
        _ = matrix_scipy @ x
        _ = matrix_spc @ x

    # 3. SciPy Timing
    scipy_times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = matrix_scipy @ x
        t1 = time.perf_counter()
        scipy_times.append((t1 - t0) * 1000.0)

    # 4. SPCraft Timing
    spc_times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = matrix_spc @ x
        t1 = time.perf_counter()
        spc_times.append((t1 - t0) * 1000.0)

    scipy_ms = float(np.median(scipy_times))
    spc_ms = float(np.median(spc_times))
    scipy_std = float(np.std(scipy_times))
    spc_std = float(np.std(spc_times))

    # 2 * nnz floating-point operations per SpMV
    flops = 2.0 * nnz
    scipy_gflops = (flops / (scipy_ms * 1e-3)) / 1e9 if scipy_ms > 0 else 0.0
    spc_gflops = (flops / (spc_ms * 1e-3)) / 1e9 if spc_ms > 0 else 0.0
    speedup = scipy_ms / spc_ms if spc_ms > 0 else 0.0

    return {
        "rows": m,
        "cols": n,
        "nnz": nnz,
        "density": nnz / (m * n) if m * n > 0 else 0.0,
        "dtype": np.dtype(dtype).name,
        "passed": passed,
        "max_err": max_err,
        "rel_err": rel_err,
        "scipy_ms": scipy_ms,
        "scipy_std": scipy_std,
        "scipy_gflops": scipy_gflops,
        "spc_ms": spc_ms,
        "spc_std": spc_std,
        "spc_gflops": spc_gflops,
        "speedup": speedup,
    }


def print_table(results: Sequence[dict[str, float | str | int | bool]]) -> None:
    """Print formatted markdown-compatible ASCII table of results."""
    header = (
        f"| {'Matrix Shape':<15} | {'NNZ':<11} | {'Dtype':<8} | "
        f"{'SciPy (ms)':<10} | {'SciPy GFLOPS':<12} | "
        f"{'SPCraft (ms)':<10} | {'SPCraft GFLOPS':<11} | {'Speedup':<8} | {'Status':<6} |"
    )
    sep = (
        f"|{'-'*17}|{'-'*13}|{'-'*10}|{'-'*12}|{'-'*14}|"
        f"{'-'*12}|{'-'*13}|{'-'*10}|{'-'*8}|"
    )
    print("\n" + header)
    print(sep)
    for r in results:
        shape_str = f"{r['rows']}x{r['cols']}"
        nnz_str = f"{r['nnz']:,}"
        status_str = "PASS" if r["passed"] else "FAIL"
        print(
            f"| {shape_str:<15} | {nnz_str:<11} | {str(r['dtype']):<8} | "
            f"{r['scipy_ms']:>10.3f} | {r['scipy_gflops']:>12.2f} | "
            f"{r['spc_ms']:>10.3f} | {r['spc_gflops']:>11.2f} | "
            f"{r['speedup']:>7.2f}x | {status_str:^6} |"
        )
    print(sep + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SpMV performance between SciPy and SPCraft."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[5_000, 10_000, 20_000, 50_000],
        help="Square matrix sizes N (matrix will be N x N).",
    )
    parser.add_argument(
        "--densities",
        nargs="+",
        type=float,
        default=[0.001],
        help="Sparsity densities (e.g. 0.001 for 0.1%% non-zeros).",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float32", "float64"],
        choices=["float32", "float64"],
        help="Data types to test.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations for timing.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--mtx",
        type=str,
        default=None,
        help="Path to Matrix Market file (.mtx) to benchmark.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to save results as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[dict[str, float | str | int | bool]] = []

    if args.mtx:
        mtx_path = Path(args.mtx)
        if not mtx_path.is_file():
            print(f"Error: Matrix file {mtx_path} not found.", file=sys.stderr)
            sys.exit(1)

        print(f"\n--- Benchmarking Matrix Market file: {mtx_path.name} ---")
        for dt_str in args.dtypes:
            dt = np.float32 if dt_str == "float32" else np.float64
            # Read via SciPy
            import scipy.io

            scipy_mat = scipy.io.mmread(str(mtx_path)).tocsr().astype(dt)
            res = run_benchmark(
                scipy_mat,
                dtype=dt,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            results.append(res)
    else:
        print("\n--- Benchmarking Synthetic Random CSR Matrices ---")
        for n in args.sizes:
            for density in args.densities:
                for dt_str in args.dtypes:
                    dt = np.float32 if dt_str == "float32" else np.float64
                    print(
                        f"Generating {n}x{n} matrix (density={density:.2%}, {dt_str})...",
                        flush=True,
                    )
                    scipy_mat = sp.random(
                        n, n, density=density, format="csr", dtype=dt, random_state=42
                    )
                    res = run_benchmark(
                        scipy_mat,
                        dtype=dt,
                        iterations=args.iterations,
                        warmup=args.warmup,
                    )
                    results.append(res)

    print_table(results)

    if args.csv:
        import csv

        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
