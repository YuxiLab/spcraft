#!/usr/bin/env python3
"""Generate identical inputs, run fresh-process LU baselines, and check accuracy."""

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess


def generate_matrix(path, shape, convection=False):
    n = 1
    for width in shape:
        n *= width
    entries = []
    dimensions = len(shape)
    for point in itertools.product(*(range(width) for width in shape)):
        row = 0
        stride = 1
        for coordinate, width in zip(point, shape):
            row += coordinate * stride
            stride *= width
        entries.append((row, row, 2.0 * dimensions + (0.25 if convection else 0.0)))
        stride = 1
        for coordinate, width in zip(point, shape):
            if coordinate > 0:
                entries.append((row, row - stride, -1.1 if convection else -1.0))
            if coordinate + 1 < width:
                entries.append((row, row + stride, -0.9 if convection else -1.0))
            stride *= width
    with path.open("w") as output:
        output.write("%%MatrixMarket matrix coordinate real general\n")
        output.write(f"{n} {n} {len(entries)}\n")
        for row, col, value in entries:
            output.write(f"{row + 1} {col + 1} {value:.17g}\n")
    return {"n": n, "nnz": len(entries), "shape": shape,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", default="1,2,4,8,16")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this workload inside a Slurm compute-node job")
    output = args.output.resolve()
    matrices = output / "matrices"
    logs = output / "raw"
    matrices.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    cases = [("poisson2d_128", (128, 128), False),
             ("poisson2d_256", (256, 256), False),
             ("convection3d_24", (24, 24, 24), True)]
    manifest = {}
    for name, shape, convection in cases:
        manifest[name] = generate_matrix(matrices / f"{name}.mtx", shape, convection)
    (output / "matrices.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for name, _, _ in cases:
        for threads in map(int, args.threads.split(",")):
            for repeat in range(args.repeats + 1):
                for solver in ("superlu_mt", "pastix"):
                    binary = "superlu_lu_baseline" if solver == "superlu_mt" else "pastix_lu_baseline"
                    command = [str(args.bin_dir.resolve() / binary),
                               "--matrix", str(matrices / f"{name}.mtx"), "--threads", str(threads)]
                    environment = dict(os.environ, OMP_NUM_THREADS=str(threads),
                                       OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close", OMP_PLACES="cores",
                                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                       BLIS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
                    log = logs / f"{name}_{solver}_t{threads}_r{repeat}.log"
                    print(f"RUN {name} {solver} threads={threads} repeat={repeat}", flush=True)
                    with log.open("w") as stream:
                        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                env=environment, timeout=180)
                    if result.returncode:
                        raise RuntimeError(f"Failed ({result.returncode}): {log}\n{log.read_text()[-4000:]}")
                    records = [line.removeprefix("SPCRAFT_BASELINE ")
                               for line in log.read_text().splitlines()
                               if line.startswith("SPCRAFT_BASELINE ")]
                    if len(records) != 1:
                        raise RuntimeError(f"Expected exactly one result in {log}")
                    row = json.loads(records[0])
                    if not row["passed"]:
                        raise RuntimeError(f"Numerical validation failed: {log}")
                    row.update(matrix=name, repeat=repeat, warmup=repeat == 0)
                    rows.append(row)
                    (output / "samples.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = []
    for name, _, _ in cases:
        for solver in ("superlu_mt", "pastix"):
            serial = [row["factor_s"] for row in rows
                      if row["matrix"] == name and row["solver"] == solver
                      and row["threads"] == 1 and not row["warmup"]]
            for threads in map(int, args.threads.split(",")):
                samples = [row for row in rows if row["matrix"] == name
                           and row["solver"] == solver and row["threads"] == threads
                           and not row["warmup"]]
                item = {"matrix": name, "solver": solver, "threads": threads,
                        "ordering": samples[0]["ordering"], "n": samples[0]["n"],
                        "nnz": samples[0]["nnz"]}
                for field in ("analyze_s", "factor_s", "solve_s", "phase_sum_s", "factor_entries"):
                    item[field] = statistics.median(row[field] for row in samples)
                item["factor_speedup"] = statistics.median(serial) / item["factor_s"]
                for field in ("backward_error", "forward_error", "relative_residual", "perturbed_pivots"):
                    item[field] = max(row[field] for row in samples)
                summary.append(item)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Sparse LU CPU baselines", "",
             "Median of three measured fresh-process runs after one discarded warmup per configuration.",
             "Both solvers use the same full CSC double-precision inputs and single-threaded OpenBLAS.",
             "SuperLU_MT: OpenMP, MMD(A^T A), partial pivoting. PaStiX: dynamic pthread scheduler, METIS, static pivoting, no compression.",
             "PaStiX binds its main thread and workers to distinct allocated cores; OpenMP uses core places and close binding.",
             "These are solver-configuration baselines; orderings and pivot strategies differ.",
             "Phase times exclude matrix loading, runtime initialization, validation, and cleanup; their sum is not process wall time.",
             "Factor-entry counts follow each library's storage accounting and are not directly interchangeable.", "",
             "| Matrix | Solver | Threads | Analyze (s) | Factor (s) | Solve (s) | Factor speedup |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['matrix']} | {row['solver']} | {row['threads']} | "
                     f"{row['analyze_s']:.6f} | {row['factor_s']:.6f} | "
                     f"{row['solve_s']:.6f} | {row['factor_speedup']:.2f} |")
    lines.extend(["", f"All {len(rows)} runs passed backward-error <= 1e-10 and forward-error <= 1e-7 checks."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"PASS: {len(rows)} validated runs; results: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
