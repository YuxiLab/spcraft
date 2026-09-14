#!/usr/bin/env python3
"""Compare the SpMV backends against one another from their raw JSON reports.

Reads ``benchmarks/results/<library>/spmv/*.json`` as written by
``benchmarks/scripts/run_spmv.sh``, and renders the thread-scaling comparison figure
plus the derived summary table into ``benchmarks/plots/figures/``.

Report names encode the machine and the template parameters they were captured
with, so a figure that would silently mix two machines or two type combinations
is refused rather than drawn: pass --machine or --types to choose one.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette; these three clear the
# all-pairs colour-vision floors, so they are safe for markers as well as lines.
# ``offset`` staggers the end-of-line value labels so the two CPU backends stay
# legible on the matrices where their curves converge.
BACKEND_STYLE = {
    "SPCraft-OpenMP": {
        "color": "#2a78d6", "marker": "o", "label": "SPCraft OpenMP", "offset": 10
    },
    "MKL": {"color": "#eb6834", "marker": "s", "label": "Intel MKL", "offset": -16},
    "cuSPARSE": {"color": "#1baf7a", "marker": "^", "label": "cuSPARSE"},
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "0.85"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("benchmarks/results"),
        help="root holding <library>/spmv/*.json reports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/plots/figures"),
        help="where the figure and its summary table are written",
    )
    parser.add_argument(
        "--machine",
        default=None,
        help="only plot reports captured on this machine tag",
    )
    parser.add_argument(
        "--types",
        default=None,
        help="only plot this type combination, e.g. 'i32/o64 FP64'",
    )
    parser.add_argument("--name", default="spmv_scaling", help="output file stem")
    return parser.parse_args()


def parse_report_name(path: Path) -> dict[str, str]:
    """Split ``spmv_<dataset>_<types>_<machine>_<resource>.json`` apart.

    Parsed from the right: a dataset name may itself contain '_' (af_shell10)
    while the three trailing fields never do.
    """
    fields = path.stem.split("_")
    if len(fields) < 5:
        raise SystemExit(
            f"{path.name}: expected spmv_<dataset>_<types>_<machine>_<resource>.json"
        )
    return {
        "kernel": fields[0],
        "dataset": "_".join(fields[1:-3]),
        "types": fields[-3],
        "machine": fields[-2],
        "resource": fields[-1],
    }


def type_label(run: dict) -> str:
    """The ``i32/o64 FP64`` label identifying a run's template parameters."""
    index = run.get("index_type", "")
    offset = run.get("offset_type", "")
    if not index or not offset:
        return run["precision"]
    return f"i{index[3:]}/o{offset[3:]} {run['precision']}"


def load_runs(results_root: Path, machine: str | None, types: str | None) -> list[dict]:
    """Flatten every run of every report into one list of records."""
    runs: list[dict] = []
    machines: set[str] = set()
    for path in sorted(results_root.glob("*/spmv/*.json")):
        tags = parse_report_name(path)
        machines.add(tags["machine"])
        # A host tag selects that host's CPU and GPU reports alike; the full
        # host-device tag selects only the GPU ones.
        if machine is not None and tags["machine"] != machine and not tags[
            "machine"
        ].startswith(machine + "-"):
            continue
        document = json.loads(path.read_text())
        for run in document["runs"]:
            run["dataset"] = run["dataset"].removesuffix(".mtx")
            run["machine"] = tags["machine"]
            run["library"] = path.parent.parent.name
            run["source"] = path.name
            run["types"] = type_label(run)
            if types is not None and run["types"] != types:
                continue
            runs.append(run)

    if not runs:
        raise SystemExit(f"no SpMV reports found under {results_root}/*/spmv/")

    # Backends legitimately differ here - oneMKL and cuSPARSE cannot represent
    # the 64-bit offsets the SPCraft kernel accepts - so the check is per
    # backend: one line per backend, and the legend names the types it used.
    # Two combinations of the *same* backend on one axis would be two different
    # kernels under one label, which is what this refuses.
    for backend in {run["backend"] for run in runs}:
        combinations = {run["types"] for run in runs if run["backend"] == backend}
        if len(combinations) > 1:
            raise SystemExit(
                f"{backend} reports span more than one type combination: "
                + ", ".join(sorted(combinations))
                + "\npass --types to pick one"
            )

    # A CPU tag and its GPU tag (host-device) describe the same box, so compare
    # against the host prefix rather than refusing every mixed CPU/GPU figure.
    hosts = {tag.split("-")[0] for tag in {run["machine"] for run in runs}}
    if len(hosts) > 1:
        raise SystemExit(
            "reports span more than one machine: "
            + ", ".join(sorted(machines))
            + "\npass --machine to pick one"
        )
    return runs


def backend_key(run: dict) -> str:
    """Map a report's backend string onto a plotting slot."""
    for key in BACKEND_STYLE:
        if key.lower() in run["backend"].lower():
            return key
    return run["backend"]


def write_csv(runs: list[dict], path: Path) -> None:
    columns = [
        "dataset", "library", "backend", "precision", "index_type", "offset_type",
        "threads", "machine",
        "rows", "columns", "nonzeros", "iterations", "median_seconds",
        "min_seconds", "stddev_seconds", "gflops", "gbytes_per_second",
        "verification_error", "source",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            writer.writerow(run)


def plot(runs: list[dict], datasets: list[str], title: str, path: Path) -> None:
    """Latency per dataset on the top row, parallel speedup on the bottom row."""
    figure, axes = plt.subplots(
        2, len(datasets), figsize=(4.1 * len(datasets), 8.0), squeeze=False
    )

    for column, dataset in enumerate(datasets):
        latency_axis = axes[0][column]
        speedup_axis = axes[1][column]
        selected = [run for run in runs if run["dataset"] == dataset]
        nonzeros = selected[0]["nonzeros"]

        for key, style in BACKEND_STYLE.items():
            series = sorted(
                (run for run in selected if backend_key(run) == key),
                key=lambda run: run["threads"],
            )
            if not series:
                continue

            if key == "cuSPARSE":
                # One measurement, no thread axis: draw it as a reference level.
                milliseconds = series[0]["median_seconds"] * 1e3
                latency_axis.axhline(
                    milliseconds,
                    color=style["color"],
                    linewidth=2,
                    linestyle="--",
                    label=f"{style['label']} ({series[0]['types']})",
                )
                latency_axis.annotate(
                    f"{milliseconds:.3f} ms",
                    xy=(1.0, milliseconds),
                    xytext=(2, 3),
                    textcoords="offset points",
                    fontsize=8,
                    color=style["color"],
                )
                continue

            threads = [run["threads"] for run in series]
            milliseconds = [run["median_seconds"] * 1e3 for run in series]
            latency_axis.plot(
                threads, milliseconds, marker=style["marker"], markersize=6,
                linewidth=2, color=style["color"],
                label=f"{style['label']} ({series[0]['types']})",
            )
            latency_axis.annotate(
                f"{milliseconds[-1]:.3f} ms",
                xy=(threads[-1], milliseconds[-1]),
                xytext=(-8, style["offset"]),
                textcoords="offset points",
                fontsize=8,
                color=style["color"],
                ha="right",
            )

            baseline = milliseconds[0]
            speedup_axis.plot(
                threads, [baseline / value for value in milliseconds],
                marker=style["marker"], markersize=6, linewidth=2,
                color=style["color"], label=style["label"],
            )

        thread_ticks = sorted({run["threads"] for run in selected if run["threads"] > 0})
        speedup_axis.plot(
            thread_ticks, thread_ticks, linestyle=":", linewidth=1.5, color="0.55",
            label="ideal",
        )

        latency_axis.set_title(
            f"{dataset}\n{nonzeros / 1e6:.2f}M nonzeros", fontsize=11, color=TEXT_PRIMARY
        )
        latency_axis.set_yscale("log")
        latency_axis.set_ylabel("Median time per SpMV (ms)" if column == 0 else "")
        # Log speedup keeps the ideal reference straight and stops a matrix that
        # does not scale from being squashed against the axis by that reference.
        speedup_axis.set_yscale("log", base=2)
        speedup_axis.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}")
        )
        speedup_axis.set_ylabel("Speedup vs 1 thread (x)" if column == 0 else "")
        speedup_axis.set_xlabel("OpenMP threads")

        for axis in (latency_axis, speedup_axis):
            axis.set_xscale("log", base=2)
            axis.set_xticks(thread_ticks)
            axis.set_xticklabels([str(count) for count in thread_ticks])
            axis.minorticks_off()
            axis.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
            axis.set_axisbelow(True)
            axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    ideal_handle, ideal_label = axes[1][0].get_legend_handles_labels()
    handles += ideal_handle[-1:]
    labels += ideal_label[-1:]
    figure.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(title, fontsize=13, color=TEXT_PRIMARY)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")


def write_markdown(runs: list[dict], datasets: list[str], path: Path,
                   figure: Path, title: str) -> None:
    lines = [f"# {title}", "", f"![SpMV thread scaling]({figure.name})", ""]
    lines.append(
        "| dataset | rows | nonzeros | backend | threads | median ms | GFLOP/s | "
        "GB/s | max error vs Eigen |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        selected = sorted(
            (run for run in runs if run["dataset"] == dataset),
            key=lambda run: (backend_key(run), run["threads"]),
        )
        for run in selected:
            lines.append(
                "| {dataset} | {rows:,} | {nonzeros:,} | {backend} | {threads} | "
                "{median:.4f} | {gflops:.2f} | {gbytes:.1f} | {error:.2e} |".format(
                    dataset=run["dataset"],
                    rows=run["rows"],
                    nonzeros=run["nonzeros"],
                    backend=run["backend"],
                    threads=run["threads"] or "-",
                    median=run["median_seconds"] * 1e3,
                    gflops=run["gflops"],
                    gbytes=run["gbytes_per_second"],
                    error=run["verification_error"],
                )
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    runs = load_runs(args.results_root, args.machine, args.types)
    datasets = sorted(
        {run["dataset"] for run in runs},
        key=lambda name: next(r["nonzeros"] for r in runs if r["dataset"] == name),
    )
    machine = sorted({run["machine"] for run in runs})[0].split("-")[0]
    title = f"CSR SpMV, {machine}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f"{args.name}.png"
    plot(runs, datasets, title, figure_path)
    write_csv(runs, args.output_dir / f"{args.name}.csv")
    write_markdown(runs, datasets, args.output_dir / f"{args.name}.md",
                   figure_path, title)
    for suffix in ("png", "csv", "md"):
        print(f"wrote {args.output_dir}/{args.name}.{suffix}")


if __name__ == "__main__":
    main()
