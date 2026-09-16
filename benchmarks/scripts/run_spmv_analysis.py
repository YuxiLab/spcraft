#!/usr/bin/env python3
"""Benchmark SPCraft and oneMKL SpMV, then plot scaling and worker balance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Dataset:
    name: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class Backend:
    key: str
    label: str
    binary: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument(
        "--real-matrix",
        type=Path,
        default=Path("dataset/output/webbase-1M/webbase-1M.mtx"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/spmv_analysis")
    )
    parser.add_argument("--threads", default="1,2,4,8,16")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--max-time", type=float, default=3.0)
    parser.add_argument("--profile-iterations", type=int, default=30)
    parser.add_argument("--cache-iterations", type=int, default=100)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    return parser.parse_args()


def metric(profile: dict, name: str) -> dict:
    return next(item for item in profile["metrics"] if item["name"] == name)


def profile(run: dict, scope: str) -> dict:
    return next(item for item in run["profiles"] if item["scope"] == scope)


def headline(series: dict) -> float:
    samples = [float(value) for value in series["samples"]]
    summary = series["headline"]
    if summary == "mean":
        return statistics.fmean(samples)
    if summary == "median":
        return statistics.median(samples)
    if summary == "min":
        return min(samples)
    if summary == "max":
        return max(samples)
    if summary == "sum":
        return sum(samples)
    if summary == "last":
        return samples[-1]
    raise ValueError(f"unknown metric summary {summary}")


def command_for(
    backend: Backend,
    dataset: Dataset,
    output: Path,
    text_output: Path,
    threads: Sequence[int],
    iterations: int,
    max_time: float,
    profile_iterations: int,
) -> list[str]:
    command = [
        str(backend.binary),
        *dataset.arguments,
        "--precision=double",
        "--index=int32",
        "--offset=int32",
        f"--threads={','.join(str(value) for value in threads)}",
        f"--iterations={iterations}",
        f"--max-time={max_time}",
        f"--output={output}",
        f"--text-report={text_output}",
    ]
    if backend.key == "spcraft":
        command += [
            f"--profile-thread-count={max(threads)}",
            f"--profile-iterations={profile_iterations}",
        ]
    return command


def run_checked(command: Sequence[str], log: Path, env: dict[str, str]) -> None:
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed; see {log}: {' '.join(command)}")


def benchmark_all(
    backends: Sequence[Backend],
    datasets: Sequence[Dataset],
    output_dir: Path,
    threads: Sequence[int],
    iterations: int,
    max_time: float,
    profile_iterations: int,
    env: dict[str, str],
) -> None:
    for dataset in datasets:
        for backend in backends:
            stem = f"{dataset.name}.{backend.key}"
            run_checked(
                command_for(
                    backend,
                    dataset,
                    output_dir / f"{stem}.json",
                    output_dir / f"{stem}.txt",
                    threads,
                    iterations,
                    max_time,
                    profile_iterations,
                ),
                output_dir / f"{stem}.log",
                env,
            )


def load_results(
    backends: Sequence[Backend], datasets: Sequence[Dataset], output_dir: Path
) -> tuple[list[dict], dict[str, list[dict]]]:
    rows: list[dict] = []
    workers: dict[str, list[dict]] = {}
    for dataset in datasets:
        for backend in backends:
            document = json.loads(
                (output_dir / f"{dataset.name}.{backend.key}.json").read_text()
            )
            for run in document["runs"]:
                cpu_profile = profile(run, "rank:0/cpu")
                rows.append(
                    {
                        "dataset": dataset.name,
                        "backend": backend.label,
                        "threads": int(run["threads"]),
                        "rows": int(run["rows"]),
                        "nonzeros": int(run["nonzeros"]),
                        "median_ms": 1000.0 * float(run["median_seconds"]),
                        "minimum_ms": 1000.0 * float(run["min_seconds"]),
                        "cv_percent": 100.0
                        * float(run["stddev_seconds"])
                        / float(run["mean_seconds"]),
                        "gflops": float(run["gflops"]),
                        "process_cpu_ms": 1000.0
                        * headline(metric(cpu_profile, "process_cpu_time")),
                        "effective_cores": headline(
                            metric(cpu_profile, "effective_cpu_cores")
                        ),
                        "cpu_occupancy": headline(
                            metric(cpu_profile, "cpu_occupancy")
                        ),
                    }
                )
                if backend.key != "spcraft" or not run["stages"]:
                    continue
                stage = next(
                    item for item in run["stages"] if item["name"] == "thread_profile"
                )
                aggregate = next(
                    item for item in stage["profiles"] if item["scope"] == "run"
                )
                workers[dataset.name] = []
                for worker in stage["profiles"]:
                    if "/thread:" not in worker["scope"]:
                        continue
                    workers[dataset.name].append(
                        {
                            "thread": int(worker["scope"].rsplit(":", 1)[1]),
                            "wall_ms": 1000.0
                            * headline(metric(worker, "thread_wall_time")),
                            "cpu_ms": 1000.0
                            * headline(metric(worker, "thread_cpu_time")),
                            "rows": int(headline(metric(worker, "assigned_rows"))),
                            "nonzeros": int(
                                headline(metric(worker, "assigned_nonzeros"))
                            ),
                            "wall_imbalance": headline(
                                metric(aggregate, "wall_time_imbalance")
                            ),
                            "cpu_imbalance": headline(
                                metric(aggregate, "cpu_time_imbalance")
                            ),
                            "nonzero_imbalance": headline(
                                metric(aggregate, "nonzero_imbalance")
                            ),
                        }
                    )
                workers[dataset.name].sort(key=lambda item: item["thread"])
    for row in rows:
        baseline = next(
            item
            for item in rows
            if item["dataset"] == row["dataset"]
            and item["backend"] == row["backend"]
            and item["threads"] == 1
        )
        row["scaling_efficiency"] = (
            float(baseline["median_ms"])
            / float(row["median_ms"])
            / int(row["threads"])
        )
    return rows, workers


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_combined(
    path: Path,
    datasets: Sequence[Dataset],
    backends: Sequence[Backend],
    rows: Sequence[dict],
    workers: dict[str, list[dict]],
    threads: Sequence[int],
) -> None:
    lookup = {
        (row["dataset"], row["backend"], row["threads"]): row for row in rows
    }
    colors = {"SPCraft": "#1f77b4", "oneMKL": "#d62728"}
    markers = {"SPCraft": "o", "oneMKL": "s"}
    figure, axes = plt.subplots(2, len(datasets), figsize=(5.2 * len(datasets), 8.5))
    for column, dataset in enumerate(datasets):
        scaling = axes[0, column]
        for backend in backends:
            baseline = float(lookup[(dataset.name, backend.label, 1)]["median_ms"])
            speedups = [
                baseline
                / float(lookup[(dataset.name, backend.label, count)]["median_ms"])
                for count in threads
            ]
            scaling.plot(
                threads,
                speedups,
                marker=markers[backend.label],
                color=colors[backend.label],
                linewidth=2,
                label=backend.label,
            )
        scaling.plot(threads, threads, "--", color="0.55", label="ideal")
        scaling.set_title(dataset.name)
        scaling.set_xscale("log", base=2)
        scaling.set_xticks(threads, labels=[str(value) for value in threads])
        scaling.set_xlabel("Threads")
        scaling.set_ylabel("Strong-scaling speedup")
        scaling.grid(True, alpha=0.25)
        if column == 0:
            scaling.legend()

        balance = axes[1, column]
        worker_rows = workers[dataset.name]
        ids = [item["thread"] for item in worker_rows]
        width = 0.4
        balance.bar(
            [value - width / 2 for value in ids],
            [item["wall_ms"] for item in worker_rows],
            width,
            color="#2ca02c",
            label="worker wall",
        )
        balance.bar(
            [value + width / 2 for value in ids],
            [item["cpu_ms"] for item in worker_rows],
            width,
            color="#9467bd",
            label="worker CPU",
        )
        balance.set_xlabel(f"SPCraft worker ({max(threads)} threads)")
        balance.set_ylabel("Time per SpMV (ms)")
        balance.set_xticks(ids)
        balance.grid(True, axis="y", alpha=0.25)
        maximum = max(threads)
        sp_row = lookup[(dataset.name, "SPCraft", maximum)]
        mk_row = lookup[(dataset.name, "oneMKL", maximum)]
        balance.set_title(
            f"wall imbalance {worker_rows[0]['wall_imbalance']:.2f}×, "
            f"NNZ {worker_rows[0]['nonzero_imbalance']:.2f}×\n"
            f"process CPU ms: SPCraft {sp_row['process_cpu_ms']:.2f}, "
            f"MKL {mk_row['process_cpu_ms']:.2f}"
        )
        if column == 0:
            balance.legend()
    figure.suptitle(
        "SpMV strong scaling and SPCraft per-worker balance\n"
        "MKL exposes process CPU work but not sparse-kernel worker timings",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def cache_analysis(
    backend: Backend,
    dataset: Dataset,
    output_dir: Path,
    threads: int,
    iterations: int,
    env: dict[str, str],
) -> dict:
    events = (
        "cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,"
        "LLC-loads,LLC-load-misses"
    )
    output = output_dir / f"cache.{backend.key}.json"
    text_output = output_dir / f"cache.{backend.key}.txt"
    driver = command_for(
        backend,
        dataset,
        output,
        text_output,
        [threads],
        iterations,
        30.0,
        1,
    )
    command = ["perf", "stat", "-x,", "-e", events, "--", *driver]
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    (output_dir / f"cache.{backend.key}.log").write_text(
        completed.stdout + completed.stderr
    )
    if completed.returncode != 0:
        error_lines = [
            line.strip()
            for line in completed.stderr.splitlines()
            if line.strip() and line.strip() != "Error:"
        ]
        return {
            "backend": backend.label,
            "available": False,
            "reason": error_lines[0] if error_lines else "perf stat failed",
        }
    counters: dict[str, int] = {}
    for line in completed.stderr.splitlines():
        fields = line.split(",")
        if len(fields) < 3 or not fields[0].strip().isdigit():
            continue
        counters[fields[2].strip()] = int(fields[0].strip())
    return {"backend": backend.label, "available": True, "counters": counters}


def write_report(
    path: Path,
    datasets: Sequence[Dataset],
    rows: Sequence[dict],
    workers: dict[str, list[dict]],
    threads: Sequence[int],
    cache: Sequence[dict],
) -> None:
    lookup = {
        (row["dataset"], row["backend"], row["threads"]): row for row in rows
    }
    maximum = max(threads)
    lines = [
        "# SpMV scaling and profiling report",
        "",
        "## Method",
        "",
        f"- Thread sweep: {', '.join(str(value) for value in threads)} with close core binding.",
        "- FP64 CSR with 32-bit column indices and row offsets; every backend result is "
        "checked against Eigen before timing.",
        "- Latency is the median of warmed-up iterations. Process CPU work is the mean "
        "because Linux can batch live-worker accounting updates.",
        "- SPCraft worker wall/CPU clocks are collected in a separate instrumented stage, "
        "so profiler overhead does not enter the primary latency.",
        "- ER and R-MAT are deterministic generated graphs; webbase-1M is loaded from "
        "`dataset/output/`.",
        "",
        f"## {maximum}-thread comparison",
        "",
        "| Dataset | SPCraft ms | oneMKL ms | SPCraft speedup | oneMKL speedup | "
        "SPCraft CPU ms | oneMKL CPU ms | wall imbalance | NNZ imbalance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        sp1 = lookup[(dataset.name, "SPCraft", 1)]
        spn = lookup[(dataset.name, "SPCraft", maximum)]
        mk1 = lookup[(dataset.name, "oneMKL", 1)]
        mkn = lookup[(dataset.name, "oneMKL", maximum)]
        worker = workers[dataset.name][0]
        lines.append(
            f"| {dataset.name} | {spn['median_ms']:.4f} | {mkn['median_ms']:.4f} | "
            f"{sp1['median_ms'] / spn['median_ms']:.2f}× | "
            f"{mk1['median_ms'] / mkn['median_ms']:.2f}× | "
            f"{spn['process_cpu_ms']:.4f} | {mkn['process_cpu_ms']:.4f} | "
            f"{worker['wall_imbalance']:.3f}× | {worker['nonzero_imbalance']:.3f}× |"
        )
    lines += ["", "## Interpretation", ""]
    for dataset in datasets:
        sp1 = lookup[(dataset.name, "SPCraft", 1)]
        spn = lookup[(dataset.name, "SPCraft", maximum)]
        mk1 = lookup[(dataset.name, "oneMKL", 1)]
        mkn = lookup[(dataset.name, "oneMKL", maximum)]
        worker = workers[dataset.name][0]
        relative = spn["median_ms"] / mkn["median_ms"]
        if worker["wall_imbalance"] < 1.1:
            balance = "SPCraft's static row assignment is balanced"
        elif abs(worker["wall_imbalance"] - worker["nonzero_imbalance"]) < 0.25:
            balance = "worker time follows the unequal nonzero assignment"
        else:
            balance = "timing imbalance is not explained by nonzero counts alone"
        lines.append(
            f"- **{dataset.name}:** SPCraft scales {sp1['median_ms'] / spn['median_ms']:.2f}× "
            f"and oneMKL {mk1['median_ms'] / mkn['median_ms']:.2f}×. At {maximum} threads "
            f"oneMKL is {relative:.2f}× faster; {balance}."
        )
    lines += [
        "",
        "MKL worker-level bars are intentionally absent: oneMKL's sparse inspector API "
        "does not expose its internal worker intervals. Its process CPU work remains in "
        "the table and figure annotations.",
        "",
        "## Cache counters",
        "",
        f"The counter experiment targets {datasets[-1].name} at {maximum} threads and "
        "requests generic cache, L1-data, and LLC load/miss events through `perf stat`.",
        "",
    ]
    for result in cache:
        if result["available"]:
            counters = result["counters"]
            lines.append(f"### {result['backend']}")
            lines.append("")
            for name, value in counters.items():
                lines.append(f"- {name}: {value:,}")
        else:
            lines.append(
                f"- {result['backend']}: unavailable ({result['reason']}). No cache-miss "
                "rate is inferred from timing data."
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    threads = [int(value) for value in args.threads.split(",")]
    if threads != sorted(set(threads)) or threads[0] != 1:
        raise ValueError("--threads must be unique, ascending, and start at 1")
    if max(threads) > (os.cpu_count() or 1):
        raise ValueError("requested thread count exceeds the available logical CPUs")
    if not args.real_matrix.is_file():
        raise FileNotFoundError(args.real_matrix)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    backends = (
        Backend("spcraft", "SPCraft", args.build_dir / "bin/bench_ompspmv"),
        Backend("mkl", "oneMKL", args.build_dir / "bin/spmv_mkl_benchmark"),
    )
    for backend in backends:
        if not backend.binary.is_file():
            raise FileNotFoundError(backend.binary)
    datasets = (
        Dataset("ER_V50000_D16", ("--generator=er", "--vertices=50000", "--degree=16")),
        Dataset("RMAT_S16_E16", ("--generator=rmat", "--scale=16", "--edge-factor=16")),
        Dataset(args.real_matrix.stem, (f"--matrix={args.real_matrix}",)),
    )
    env = os.environ.copy()
    env.update(
        {
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "OMP_WAIT_POLICY": "active",
            "MKL_DYNAMIC": "FALSE",
        }
    )
    if not args.report_only:
        benchmark_all(
            backends,
            datasets,
            output_dir,
            threads,
            args.iterations,
            args.max_time,
            args.profile_iterations,
            env,
        )

    rows, workers = load_results(backends, datasets, output_dir)
    write_csv(output_dir / "summary.csv", rows)
    worker_rows = [
        {"dataset": dataset, **worker}
        for dataset, entries in workers.items()
        for worker in entries
    ]
    write_csv(output_dir / "threads.csv", worker_rows)
    plot_combined(
        output_dir / "spmv_analysis.png", datasets, backends, rows, workers, threads
    )

    cache: list[dict] = []
    if not args.skip_cache and not args.report_only:
        if shutil.which("perf") is None:
            cache = [
                {"backend": backend.label, "available": False, "reason": "perf not installed"}
                for backend in backends
            ]
        else:
            cache = [
                cache_analysis(
                    backend,
                    datasets[-1],
                    output_dir,
                    max(threads),
                    args.cache_iterations,
                    env,
                )
                for backend in backends
            ]
        (output_dir / "cache.json").write_text(json.dumps(cache, indent=2) + "\n")
    elif (output_dir / "cache.json").is_file():
        cache = json.loads((output_dir / "cache.json").read_text())
    else:
        cache = [
            {"backend": backend.label, "available": False, "reason": "cache stage skipped"}
            for backend in backends
        ]
    write_report(output_dir / "report.md", datasets, rows, workers, threads, cache)
    print(f"figure: {output_dir / 'spmv_analysis.png'}")
    print(f"report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
