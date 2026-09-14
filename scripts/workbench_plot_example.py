#!/usr/bin/env python3
"""
Custom Downstream Python Visualization Script for CMake Benchmark Workbench.

This script is invoked automatically by the CMake Benchmark Workbench after a binary runs.
It receives:
  1. --data <path_to_json>: Path to JSON file containing execution results, metrics, and iteration timings.
  2. --plot-dir <dir>: Directory where plot images should be saved.
  3. stdin: Also receives the raw JSON payload via standard input.

Any image saved in --plot-dir (or printed as `PLOT_IMAGE: <path>`) will be automatically
loaded and displayed in the Workbench's "Custom Python Visualizations" gallery.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Workbench Downstream Custom Plotter")
    parser.add_argument("--data", type=str, help="Path to JSON run data", default=None)
    parser.add_argument("--plot-dir", type=str, help="Output plot directory", default=".bench_plots")
    parser.add_argument("--dpi", type=int, help="Plot resolution DPI", default=150)
    parser.add_argument("--style", type=str, help="Matplotlib style", default="default")
    args, unknown = parser.parse_known_args()

    # 1. Load run data from --data file or stdin
    if args.data and os.path.exists(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            run_data = json.load(f)
    elif not sys.stdin.isatty():
        run_data = json.load(sys.stdin)
    else:
        print("[Python Downstream] Error: No input data provided via --data or stdin.", file=sys.stderr)
        sys.exit(1)

    target_name = run_data.get("targetName", "Unknown Target")
    dataset_name = run_data.get("datasetName") or "Default Dataset"
    duration_ms = run_data.get("durationMs", 0)
    metrics = run_data.get("metrics", [])
    iteration_timings = run_data.get("iterationTimings", [])

    print(f"[Python Downstream] Processing results for target: {target_name} on {dataset_name} ({duration_ms} ms)")

    # 2. Setup output plot directory
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create Custom Visualizations with Matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless backend
        import matplotlib.pyplot as plt
        import numpy as np

        # Styling
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

        # Plot 1: Per-Iteration Latency & Moving Average (or Synthetic Latency breakdown)
        ax1 = axes[0]
        if iteration_timings and len(iteration_timings) > 1:
            iters = np.arange(1, len(iteration_timings) + 1)
            y = np.array(iteration_timings)
            ax1.plot(iters, y, "o-", color="#4ec9b0", lw=2, markersize=5, label="Iteration Latency (ms)")

            # Moving average
            if len(y) >= 4:
                window = min(3, len(y))
                ma = np.convolve(y, np.ones(window) / window, mode="valid")
                ax1.plot(iters[window - 1:], ma, "--", color="#cca700", lw=2, label=f"MA ({window}-iter)")

            ax1.axhline(np.median(y), color="#f14c4c", linestyle=":", lw=1.5, label=f"Median: {np.median(y):.3f} ms")
            ax1.set_xlabel("Iteration Number", fontsize=11)
            ax1.set_ylabel("Latency (ms)", fontsize=11)
            ax1.set_title(f"Iteration Profile: {target_name}", fontsize=12, fontweight="bold")
            ax1.legend(loc="upper right", frameon=True)
        else:
            # Fallback when only total time is present
            ax1.bar(["Total Runtime"], [duration_ms], color="#3794ff", width=0.4)
            ax1.set_ylabel("Time (ms)", fontsize=11)
            ax1.set_title(f"Runtime: {target_name}", fontsize=12, fontweight="bold")
            ax1.grid(True, linestyle="--", alpha=0.6)

        # Plot 2: Extracted Metrics Bar Chart
        ax2 = axes[1]
        metric_names = [m["name"] for m in metrics] if metrics else ["Duration"]
        metric_vals = [m["value"] for m in metrics] if metrics else [duration_ms]
        metric_units = [m.get("unit", "") for m in metrics] if metrics else ["ms"]
        labels = [f"{n}\n({u})" if u else n for n, u in zip(metric_names, metric_units)]

        colors = ["#4ec9b0", "#3794ff", "#bc3fbc", "#cca700", "#4fc1ff"]
        bar_colors = [colors[i % len(colors)] for i in range(len(metric_vals))]

        bars = ax2.bar(labels, metric_vals, color=bar_colors, width=0.55, edgecolor="#ffffff", linewidth=0.5)
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
            )

        ax2.set_title(f"Performance Metrics ({dataset_name})", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Metric Value", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

        # Save Output Figure
        output_image_path = plot_dir / f"{target_name}_custom_analysis.png"
        fig.savefig(output_image_path, dpi=args.dpi)
        plt.close(fig)

        # Inform the Workbench that this image is ready for display!
        print(f"PLOT_IMAGE: {output_image_path.resolve()}")
        print(f"[Python Downstream] Custom figure successfully generated at: {output_image_path}")

    except Exception as ex:
        print(f"[Python Downstream] Error during plotting: {ex}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
