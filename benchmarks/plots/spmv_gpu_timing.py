#!/usr/bin/env python3
"""Scatter cuSPARSE SpMV timing against matrix size.

Reads the raw JSON reports written by ``benchmarks/scripts/run_spmv.sh`` for the
cusparse driver and plots per-matrix median time (ms) against log10(nnz), one
series per template combination (index/offset width x precision).

Usage:
  ./benchmarks/plots/spmv_gpu_timing.py [--results-root benchmarks/results] \\
      [--output benchmarks/plots/figures/gpu_spmv_timing.png] [--show-legend]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES_STYLE = {
    ("i32-o32", "FP32"): {"color": "#2a78d6", "marker": "o", "label": "i32-o32 FP32"},
    ("i32-o32", "FP64"): {"color": "#1baf7a", "marker": "^", "label": "i32-o32 FP64"},
    ("i64-o64", "FP32"): {"color": "#eb6834", "marker": "s", "label": "i64-o64 FP32"},
    ("i64-o64", "FP64"): {"color": "#8a4bd6", "marker": "D", "label": "i64-o64 FP64"},
}
TEXT_SECONDARY = "#52514e"
GRID = "0.85"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("benchmarks/results"),
                        help="root holding <library>/spmv/*.json reports")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/plots/figures/gpu_spmv_timing.png"))
    parser.add_argument("--machine", default=None,
                        help="only plot reports captured on this machine tag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    series: dict[tuple[str, str], dict[str, list[float]]] = {
        key: {"nnz": [], "ms": []} for key in SERIES_STYLE
    }
    machines: set[str] = set()
    for path in sorted(args.results_root.glob("*/spmv/*.json")):
        tags = path.stem.split("_")
        if len(tags) < 5:
            continue
        machine = tags[-2]
        machines.add(machine)
        if args.machine and machine != args.machine:
            continue
        types = tags[-3]  # e.g. i32-o32-fpall
        if not (types.startswith("i32-o32") or types.startswith("i64-o64")):
            continue
        for run in json.loads(path.read_text()).get("runs", []):
            precision = run.get("precision", "")
            index = run.get("index_type", "")
            offset = run.get("offset_type", "")
            key = (f"i{index[3:]}-o{offset[3:]}", precision)
            if key not in series:
                continue
            series[key]["nnz"].append(run.get("nonzeros", 0))
            series[key]["ms"].append(run.get("median_seconds", float("nan")) * 1.0e3)

    if args.machine and machines and args.machine not in machines:
        print(f"warning: no reports matched machine tag '{args.machine}' (found: {machines})")

    figure, axis = plt.subplots(figsize=(9, 6))
    for (combo, precision), style in SERIES_STYLE.items():
        data = series[(combo, precision)]
        if not data["nnz"]:
            print(f"no data for {combo} {precision}")
            continue
        nnz = np.asarray(data["nnz"])
        ms = np.asarray(data["ms"])
        order = np.argsort(nnz)
        axis.scatter(np.log10(nnz[order]), ms[order], s=8, alpha=0.55,
                     color=style["color"], marker=style["marker"],
                     label=style["label"], linewidths=0)

    axis.set_xlabel("lg(nnz)")
    axis.set_ylabel("median SpMV time (ms)")
    axis.set_yscale("log")
    axis.grid(True, which="both", color=GRID, linewidth=0.5)
    axis.legend(frameon=False, fontsize=9)
    title = "cuSPARSE SpMV timing, SuiteSparse"
    if args.machine:
        title += f" ({args.machine})"
    axis.set_title(title, color=TEXT_SECONDARY, fontsize=11)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
