#!/usr/bin/env python3
"""Run the ER-graph SpMV benchmark and plot latency and scaling."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CONFIGS = ((100_000, 100), (500_000, 30), (1_000_000, 15))
DEFAULT_THREADS = (1, 2, 4, 8, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path, help="path to the spmv_er_openmp executable")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--expected-degree", type=float, default=16.0)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for vertices, iterations in DEFAULT_CONFIGS:
        command = [
            str(args.benchmark.resolve()),
            str(vertices),
            str(args.expected_degree),
            str(iterations),
            str(args.samples),
            str(args.seed),
            *(str(value) for value in DEFAULT_THREADS),
        ]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        rows.extend(csv.DictReader(io.StringIO(completed.stdout)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "spmv_er_openmp.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    figure, (latency_axis, speedup_axis) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = plt.cm.viridis([0.15, 0.52, 0.85])
    for color, (vertices, _) in zip(colors, DEFAULT_CONFIGS):
        selected = [row for row in rows if int(row["vertices"]) == vertices]
        threads = [int(row["threads"]) for row in selected]
        latency = [float(row["median_ms"]) for row in selected]
        baseline = latency[0]
        label = f"{vertices / 1_000_000:g}M vertices"
        latency_axis.plot(threads, latency, marker="o", linewidth=2, color=color, label=label)
        speedup_axis.plot(
            threads,
            [baseline / value for value in latency],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

    speedup_axis.plot(DEFAULT_THREADS, DEFAULT_THREADS, "--", color="0.55", label="ideal")
    latency_axis.set_title("SpMV latency")
    latency_axis.set_xlabel("OpenMP threads")
    latency_axis.set_ylabel("Median time per SpMV (ms)")
    latency_axis.set_yscale("log")
    speedup_axis.set_title("Speedup relative to one thread")
    speedup_axis.set_xlabel("OpenMP threads")
    speedup_axis.set_ylabel("Speedup (×)")
    for axis in (latency_axis, speedup_axis):
        axis.set_xticks(DEFAULT_THREADS)
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.suptitle(f"CSR SpMV on Erdős–Rényi G(n,p), expected degree {args.expected_degree:g}")
    figure.tight_layout()
    figure.savefig(args.output_dir / "spmv_er_openmp.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
