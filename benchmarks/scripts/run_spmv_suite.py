#!/usr/bin/env python3
"""Download and benchmark a representative ten-matrix SpMV suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatrixSpec:
    group: str
    name: str
    category: str


#: Backend names as the drivers write them, mapped to the short report labels.
BACKEND_LABELS = {"SPCraft-OpenMP": "SPCraft", "oneMKL": "MKL"}


MATRICES = (
    MatrixSpec("McRae", "ecology1", "2D/3D discretization"),
    MatrixSpec("SNAP", "roadNet-CA", "road graph"),
    MatrixSpec("Williams", "webbase-1M", "web graph"),
    MatrixSpec("SNAP", "amazon0312", "product graph"),
    MatrixSpec("AMD", "G3_circuit", "circuit simulation"),
    MatrixSpec("Schmid", "thermal2", "thermal FEM"),
    MatrixSpec("Rothberg", "cfd2", "computational fluid dynamics"),
    MatrixSpec("Williams", "cant", "structural mechanics"),
    MatrixSpec("Williams", "cop20k_A", "finite-element model"),
    MatrixSpec("Schenk_AFE", "af_shell10", "large structural shell"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spcraft-benchmark",
        type=Path,
        default=Path("build/benchmarks/spcraft/spmv_openmp_benchmark"),
    )
    parser.add_argument(
        "--mkl-benchmark",
        type=Path,
        default=Path("build/benchmarks/mkl/spmv_mkl_benchmark"),
    )
    parser.add_argument(
        "--profiler",
        type=Path,
        default=Path("build/benchmarks/spcraft/spmv_openmp_thread_profile"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset/suitsparse"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/spmv_mkl")
    )
    parser.add_argument("--threads", default="1,2,4,8,16")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--max-time", type=float, default=10.0)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def matrix_path(data_root: Path, spec: MatrixSpec) -> Path:
    return data_root / spec.name / f"{spec.name}.mtx"


def download_matrix(data_root: Path, spec: MatrixSpec) -> Path:
    target = matrix_path(data_root, spec)
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://sparse-files.engr.tamu.edu/MM/{spec.group}/{spec.name}.tar.gz"
    archive = target.parent / f"{spec.name}.tar.gz.part"
    print(f"Downloading {spec.name}: {url}", flush=True)
    with urllib.request.urlopen(url) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)

    wanted_name = f"{spec.name}.mtx"
    with tarfile.open(archive, "r:gz") as bundle:
        member = next(
            (
                entry
                for entry in bundle.getmembers()
                if entry.isfile() and Path(entry.name).name == wanted_name
            ),
            None,
        )
        if member is None:
            raise RuntimeError(f"{wanted_name} is missing from {url}")
        source = bundle.extractfile(member)
        if source is None:
            raise RuntimeError(f"cannot extract {member.name} from {url}")
        with source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    archive.unlink()
    return target


def run_command(
    command: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, capture_output=True, env=env)


def read_benchmark(path: Path, category: str) -> list[dict[str, object]]:
    """Read a BenchmarkReport JSON log into flat rows.

    Each run already carries its per-iteration timings, so the coefficient of
    variation comes straight from the recorded mean and standard deviation
    rather than from a separate aggregate entry.
    """
    document = json.loads(path.read_text())
    rows: list[dict[str, object]] = []
    for entry in document["runs"]:
        mean = float(entry["mean_seconds"])
        rows.append(
            {
                "matrix": entry["dataset"].removesuffix(".mtx"),
                "category": category,
                "backend": BACKEND_LABELS.get(entry["backend"], entry["backend"]),
                "precision": entry["precision"],
                "threads": int(entry["threads"]),
                "rows": int(entry["rows"]),
                "cols": int(entry["columns"]),
                "nnz": int(entry["nonzeros"]),
                "median_ms": float(entry["median_seconds"]) * 1.0e3,
                "gflops": float(entry["gflops"]),
                "useful_gb_s": float(entry["gbytes_per_second"]),
                "cv_percent": (
                    100.0 * float(entry["stddev_seconds"]) / mean
                    if mean > 0.0
                    else math.nan
                ),
            }
        )
    return rows


SUMMARY_PATTERN = re.compile(
    r"matrix=(?P<matrix>\S+).*?mean_work_ms=(?P<mean>[0-9.eE+-]+).*?"
    r"max_work_ms=(?P<maximum>[0-9.eE+-]+).*?time_imbalance=(?P<time>[0-9.eE+-]+).*?"
    r"nnz_imbalance=(?P<nnz>[0-9.eE+-]+)"
)


def run_profile(
    profiler: Path,
    matrix: Path,
    threads: int,
    iterations: int,
    output_dir: Path,
    env: dict[str, str],
) -> dict[str, object]:
    completed = run_command(
        [
            str(profiler),
            f"--matrix={matrix}",
            f"--threads={threads}",
            f"--iterations={iterations}",
        ],
        env,
    )
    header = "matrix,thread,cpu_start,cpu_end,rows,nnz,avg_work_ms,estimated_wait_ms"
    header_offset = completed.stdout.find(header)
    if header_offset < 0:
        raise RuntimeError(f"thread-profile CSV header missing for {matrix}")
    (output_dir / f"{matrix.stem}_threads.csv").write_text(
        completed.stdout[header_offset:]
    )
    match = SUMMARY_PATTERN.search(completed.stderr)
    if match is None:
        raise RuntimeError(
            f"thread-profile summary missing for {matrix}: {completed.stderr}"
        )
    return {
        "matrix": matrix.stem,
        "mean_work_ms": float(match["mean"]),
        "max_work_ms": float(match["maximum"]),
        "time_imbalance": float(match["time"]),
        "nnz_imbalance": float(match["nnz"]),
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_report(
    output_dir: Path,
    results: list[dict[str, object]],
    profiles: list[dict[str, object]],
    thread_counts: list[int],
    iterations: int,
    max_time: float,
) -> None:
    lookup = {(row["matrix"], row["backend"], row["threads"]): row for row in results}
    profile_lookup = {row["matrix"]: row for row in profiles}
    max_threads = max(thread_counts)
    comparison_threads = 8 if 8 in thread_counts else max_threads

    ratios: list[float] = []
    spcraft_scaling: list[float] = []
    mkl_scaling: list[float] = []
    comparison_rows: list[str] = []
    matrix_order = [spec.name for spec in MATRICES]
    for name in matrix_order:
        sp1 = lookup[(name, "SPCraft", 1)]
        spn = lookup[(name, "SPCraft", comparison_threads)]
        mk1 = lookup[(name, "MKL", 1)]
        mkn = lookup[(name, "MKL", comparison_threads)]
        profile = profile_lookup[name]
        ratio = float(spn["median_ms"]) / float(mkn["median_ms"])
        sp_scale = float(sp1["median_ms"]) / float(spn["median_ms"])
        mk_scale = float(mk1["median_ms"]) / float(mkn["median_ms"])
        ratios.append(ratio)
        spcraft_scaling.append(sp_scale)
        mkl_scaling.append(mk_scale)
        comparison_rows.append(
            f"| {name} | {int(spn['rows']):,} | {int(spn['nnz']):,} | "
            f"{float(spn['median_ms']):.3f} | {float(spn['gflops']):.2f} | "
            f"{float(mkn['median_ms']):.3f} | {float(mkn['gflops']):.2f} | "
            f"{ratio:.2f}× | {sp_scale:.2f}× | {float(profile['time_imbalance']):.2f}× | "
            f"{float(profile['nnz_imbalance']):.2f}× |"
        )

    scaling_rows: list[str] = []
    for threads in thread_counts:
        sp_values = [
            float(lookup[(name, "SPCraft", 1)]["median_ms"])
            / float(lookup[(name, "SPCraft", threads)]["median_ms"])
            for name in matrix_order
        ]
        mk_values = [
            float(lookup[(name, "MKL", 1)]["median_ms"])
            / float(lookup[(name, "MKL", threads)]["median_ms"])
            for name in matrix_order
        ]
        scaling_rows.append(
            f"| {threads} | {geometric_mean(sp_values):.2f}× | {geometric_mean(mk_values):.2f}× |"
        )

    mkl_wins = sum(ratio > 1.0 for ratio in ratios)
    best_mkl = matrix_order[ratios.index(max(ratios))]
    best_spcraft = matrix_order[ratios.index(min(ratios))]
    comparison_cvs = [
        float(row["cv_percent"])
        for row in results
        if int(row["threads"]) == comparison_threads
    ]
    maximum_thread_cvs = [
        float(row["cv_percent"])
        for row in results
        if int(row["threads"]) == max_threads
    ]

    # When the comparison and maximum thread counts coincide there is no second
    # population to contrast, so the noise paragraph collapses to one sentence.
    if comparison_threads == max_threads:
        stability = (
            f"The {comparison_threads}-thread comparison has median CV "
            f"{statistics.median(comparison_cvs):.2f}% and worst CV "
            f"{max(comparison_cvs):.2f}% on this shared VM."
        )
    else:
        stability = (
            f"The {comparison_threads}-thread comparison is the primary result because its "
            f"median CV is {statistics.median(comparison_cvs):.2f}% and its worst CV is "
            f"{max(comparison_cvs):.2f}%. At {max_threads} threads this shared VM is visibly "
            f"noisier: median CV {statistics.median(maximum_thread_cvs):.2f}%, worst CV "
            f"{max(maximum_thread_cvs):.2f}%. The {max_threads}-thread entry in the scaling "
            "table is useful as a saturation indicator, but small differences there should "
            "not be treated as conclusive."
        )

    def ratio_for(name: str) -> float:
        return float(
            lookup[(name, "SPCraft", comparison_threads)]["median_ms"]
        ) / float(lookup[(name, "MKL", comparison_threads)]["median_ms"])

    web_profile = profile_lookup["webbase-1M"]
    cop_profile = profile_lookup["cop20k_A"]
    g3_profile = profile_lookup["G3_circuit"]
    shell_profile = profile_lookup["af_shell10"]
    shell_scaling = float(lookup[("af_shell10", "SPCraft", 1)]["median_ms"]) / float(
        lookup[("af_shell10", "SPCraft", comparison_threads)]["median_ms"]
    )
    report = f"""# SPCraft OpenMP vs oneMKL SpMV performance

## Method

- CPU: 16-core AMD EPYC Milan virtual machine, one hardware thread per core.
- Build: `-O3 -DNDEBUG`, GCC, oneMKL 2026.1 GNU OpenMP LP64 backend.
- Precision: FP64; CSR uses 32-bit row offsets and column indices for both implementations.
- Affinity: `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `OMP_WAIT_POLICY=active`.
- Timing: SPCraft's own loop records every iteration; the median is reported. Up to {iterations} iterations per configuration within a {max_time:.1f} s budget.
- Each configuration is warmed up at its exact thread count before timing, so OpenMP team creation is not measured.
- Both implementations are verified against an Eigen sparse product before timing.
- oneMKL CSR handle creation, hints, and `mkl_sparse_optimize` are outside the timed loop.
- Each MKL thread count has a separately optimized handle because its inspector partitions work for the active team size.
- `useful_gb_s` is a modeled useful-byte rate, not a hardware DRAM counter.
- Per-thread imbalance was sampled with {max_threads} pinned OpenMP threads over 20 iterations.

## {comparison_threads}-thread comparison

`MKL/SPCraft` is `SPCraft time / MKL time`: above 1 means MKL is faster.

| Matrix | Rows | NNZ | SPCraft ms | SPCraft GF/s | MKL ms | MKL GF/s | MKL/SPCraft | SPCraft scaling | Time imbalance | NNZ imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

## Aggregate findings

- oneMKL wins on {mkl_wins}/{len(ratios)} matrices at {comparison_threads} threads.
- Geometric-mean MKL speed relative to SPCraft: **{geometric_mean(ratios):.2f}×**.
- Geometric-mean {comparison_threads}-thread scaling: SPCraft **{geometric_mean(spcraft_scaling):.2f}×**, MKL **{geometric_mean(mkl_scaling):.2f}×**.
- MKL's largest advantage is on `{best_mkl}` ({max(ratios):.2f}×); SPCraft's strongest relative result is on `{best_spcraft}` ({1.0 / min(ratios):.2f}× faster than MKL).
- Per-thread time imbalance is `max(thread work time) / mean(thread work time)`. NNZ imbalance uses the same ratio for assigned nonzeros. A value of 1 is perfectly balanced.

## Measurement stability

{stability}

## Geometric-mean scaling across the suite

| Threads | SPCraft speedup | MKL speedup |
|---:|---:|---:|
{chr(10).join(scaling_rows)}

## Interpretation

SPCraft statically divides rows, while row costs track nonzeros and irregular `x` accesses. Matrices with high NNZ or time imbalance leave early-finishing threads waiting at the implicit OpenMP barrier. Compare each matrix's `*_threads.csv` file to determine whether its limitation is uneven work assignment (high NNZ imbalance) or per-nonzero/cache behavior (time imbalance materially above NNZ imbalance).

- `webbase-1M` and `cop20k_A` expose row-count scheduling imbalance: their NNZ imbalances are {float(web_profile['nnz_imbalance']):.2f}× and {float(cop_profile['nnz_imbalance']):.2f}×, closely tracking time imbalances of {float(web_profile['time_imbalance']):.2f}× and {float(cop_profile['time_imbalance']):.2f}×. MKL is {ratio_for('webbase-1M'):.2f}× and {ratio_for('cop20k_A'):.2f}× faster, respectively.
- `amazon0312` and `roadNet-CA` favor SPCraft's lower-overhead kernel: SPCraft is {1.0 / ratio_for('amazon0312'):.2f}× and {1.0 / ratio_for('roadNet-CA'):.2f}× faster, respectively.
- `G3_circuit` has only {float(g3_profile['nnz_imbalance']):.2f}× NNZ imbalance but {float(g3_profile['time_imbalance']):.2f}× time imbalance, pointing to irregular vector access or cache locality rather than only unequal nonzero counts.
- `af_shell10` has nearly perfect NNZ balance ({float(shell_profile['nnz_imbalance']):.2f}×), yet {float(shell_profile['time_imbalance']):.2f}× time imbalance and only {shell_scaling:.2f}× SPCraft scaling at {comparison_threads} threads. Its limit is therefore bandwidth/locality or VM interference, not row-count balance.

The clearest next experiment is an NNZ-balanced static row partition precomputed from `row_ptr`. It retains contiguous row ownership while directly targeting the imbalance seen on `webbase-1M` and `cop20k_A`; a dynamic OpenMP schedule is also worth measuring, but may add scheduling overhead and hurt locality.

The benchmark measures steady-state repeated SpMV with hot allocations. It does not include Matrix Market parsing, CSR construction, or oneMKL inspector setup. Results are specific to this AMD EPYC virtual machine and should not be generalized to Intel CPUs without rerunning the suite.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {spec.name: download_matrix(data_root, spec) for spec in MATRICES}
    if args.download_only:
        return

    thread_counts = [int(value) for value in args.threads.split(",")]
    if args.report_only:
        with (output_dir / "results.csv").open() as source:
            results = list(csv.DictReader(source))
        for row in results:
            row["threads"] = int(row["threads"])
        with (output_dir / "thread_summary.csv").open() as source:
            profiles = list(csv.DictReader(source))
        generate_report(
            output_dir, results, profiles, thread_counts, args.iterations, args.max_time
        )
        print(f"Report: {output_dir / 'report.md'}")
        return

    benchmarks = {
        "spcraft": args.spcraft_benchmark.resolve(),
        "mkl": args.mkl_benchmark.resolve(),
    }
    profiler = args.profiler.resolve()
    for label, path in {**benchmarks, "profiler": profiler}.items():
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{label} binary not found at {path}; build with "
                "-DSPCRAFT_BUILD_BENCHMARKS=ON -DSPCRAFT_USE_MKL=ON"
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

    results: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    for spec in MATRICES:
        matrix = paths[spec.name]
        # SPCraft and oneMKL are separate binaries now, so each writes its own
        # log and the two are concatenated into one result set.
        for label, benchmark in benchmarks.items():
            json_path = output_dir / f"{spec.name}.{label}.json"
            completed = run_command(
                [
                    str(benchmark),
                    f"--matrix={matrix}",
                    "--precision=double",
                    f"--threads={args.threads}",
                    f"--iterations={args.iterations}",
                    f"--max-time={args.max_time}",
                    f"--output={json_path}",
                ],
                env,
            )
            (output_dir / f"{spec.name}.{label}.log").write_text(
                completed.stdout + completed.stderr
            )
            results.extend(read_benchmark(json_path, spec.category))
        profiles.append(
            run_profile(
                profiler,
                matrix,
                max(thread_counts),
                args.profile_iterations,
                output_dir,
                env,
            )
        )

    for row in results:
        baseline = next(
            candidate
            for candidate in results
            if candidate["matrix"] == row["matrix"]
            and candidate["backend"] == row["backend"]
            and candidate["threads"] == 1
        )
        row["speedup"] = float(baseline["median_ms"]) / float(row["median_ms"])

    write_csv(output_dir / "results.csv", results)
    write_csv(output_dir / "thread_summary.csv", profiles)
    generate_report(
        output_dir, results, profiles, thread_counts, args.iterations, args.max_time
    )
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
