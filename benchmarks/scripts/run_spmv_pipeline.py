#!/usr/bin/env python3
"""End-to-end SpMV benchmark pipeline: dataset, correctness, timing, figure.

Five stages, each of which can be run alone with ``--stage``:

1. ``dataset``   confirm the ten-matrix suite is present, downloading what is not,
                 and read back each header so a truncated file is caught here
                 rather than three hours later.
2. ``verify``    run every backend over every matrix for a handful of iterations
                 and require each one to agree with the Eigen reference the
                 drivers check themselves against. A backend that fails here
                 stops the pipeline: timing a wrong kernel is worse than useless.
3. ``benchmark`` the timed run, writing one JSON log per backend and matrix.
4. ``figure``    a publication-quality comparison figure plus the table behind it.

Execution happens either on this machine or through Slurm. ``--executor auto``
submits a batch job when ``sbatch`` exists and we are not already inside an
allocation -- that is, on a login node -- and runs locally otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixSpec:
    group: str
    name: str
    category: str


#: The ten matrices, spanning the sparsity patterns SpMV behaves differently on:
#: banded discretizations, power-law graphs, and dense-ish FEM blocks.
MATRICES: tuple[MatrixSpec, ...] = (
    MatrixSpec("Williams", "cant", "structural mechanics"),
    MatrixSpec("Williams", "cop20k_A", "finite-element model"),
    MatrixSpec("Rothberg", "cfd2", "computational fluid dynamics"),
    MatrixSpec("McRae", "ecology1", "2D/3D discretization"),
    MatrixSpec("AMD", "G3_circuit", "circuit simulation"),
    MatrixSpec("Schmid", "thermal2", "thermal FEM"),
    MatrixSpec("Schenk_AFE", "af_shell10", "large structural shell"),
    MatrixSpec("SNAP", "amazon0312", "product graph"),
    MatrixSpec("SNAP", "roadNet-CA", "road graph"),
    MatrixSpec("Williams", "webbase-1M", "web graph"),
)


@dataclass(frozen=True)
class Backend:
    key: str
    label: str
    #: Path of the driver relative to the CMake build directory.
    binary: str
    device: str
    #: oneMKL's CSR is MKL_INT on both index and offset, so it cannot run int64.
    offset: str


BACKENDS: tuple[Backend, ...] = (
    Backend("spcraft", "SPCraft", "benchmarks/spcraft/spmv_openmp_benchmark", "cpu", "int32"),
    Backend("mkl", "oneMKL", "benchmarks/mkl/spmv_mkl_benchmark", "cpu", "int32"),
    Backend("cusparse", "cuSPARSE", "benchmarks/cusparse/cusparse_spmv_benchmark", "gpu", "int32"),
)

#: A kernel whose result drifts further than this from Eigen is not worth timing.
VERIFICATION_TOLERANCE = 1e-6


# --------------------------------------------------------------------------
# Stage 1: dataset
# --------------------------------------------------------------------------


def matrix_path(data_root: Path, spec: MatrixSpec) -> Path:
    return data_root / spec.name / f"{spec.name}.mtx"


def read_mtx_header(path: Path) -> tuple[int, int, int]:
    """Rows, columns and stored entries from the Matrix Market banner."""
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if line.startswith("%"):
                continue
            if not line.strip():
                continue
            rows, columns, entries = (int(field) for field in line.split()[:3])
            return rows, columns, entries
    raise RuntimeError(f"{path} has no Matrix Market dimension line")


def download_matrix(data_root: Path, spec: MatrixSpec) -> Path:
    target = matrix_path(data_root, spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://sparse-files.engr.tamu.edu/MM/{spec.group}/{spec.name}.tar.gz"
    # Download beside the target, then move: an interrupted run must not leave a
    # half-written .mtx that looks present to the next invocation.
    archive = target.parent / f"{spec.name}.tar.gz.part"
    print(f"  downloading {spec.name} from {url}", flush=True)
    with urllib.request.urlopen(url) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output, length=1 << 20)

    wanted = f"{spec.name}.mtx"
    with tarfile.open(archive, "r:gz") as bundle:
        member = next(
            (e for e in bundle.getmembers() if e.isfile() and Path(e.name).name == wanted),
            None,
        )
        if member is None:
            raise RuntimeError(f"{wanted} is missing from {url}")
        source = bundle.extractfile(member)
        if source is None:
            raise RuntimeError(f"cannot extract {member.name} from {url}")
        partial = target.with_suffix(".mtx.part")
        with source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=1 << 20)
        partial.replace(target)
    archive.unlink()
    return target


def stage_dataset(data_root: Path, specs: Sequence[MatrixSpec], allow_download: bool) -> list[dict]:
    print(f"[dataset] confirming {len(specs)} matrices under {data_root}")
    summary: list[dict] = []
    for spec in specs:
        path = matrix_path(data_root, spec)
        if not path.is_file():
            if not allow_download:
                raise SystemExit(
                    f"{path} is missing and --no-download was given; "
                    f"run with downloads enabled or populate {data_root}"
                )
            path = download_matrix(data_root, spec)
        rows, columns, entries = read_mtx_header(path)
        size_mb = path.stat().st_size / (1 << 20)
        print(
            f"  {spec.name:<14} {rows:>9,} x {columns:<9,} entries {entries:>11,}"
            f"  {size_mb:7.1f} MB  ({spec.category})"
        )
        summary.append(
            {
                "name": spec.name,
                "group": spec.group,
                "category": spec.category,
                "path": str(path),
                "rows": rows,
                "columns": columns,
                "stored_entries": entries,
            }
        )
    return summary


# --------------------------------------------------------------------------
# Stage 2/3: running the drivers
# --------------------------------------------------------------------------


def driver_path(build_dir: Path, backend: Backend) -> Path:
    return build_dir / backend.binary


def available_backends(build_dir: Path, wanted: Iterable[str]) -> list[Backend]:
    """The requested backends whose driver was actually built."""
    found = []
    for backend in BACKENDS:
        if backend.key not in wanted:
            continue
        path = driver_path(build_dir, backend)
        if path.is_file() and os.access(path, os.X_OK):
            found.append(backend)
        else:
            print(f"  ! {backend.label} driver not built at {path}; skipping")
    return found


def driver_command(
    build_dir: Path,
    backend: Backend,
    matrix: Path,
    output: Path,
    *,
    threads: str,
    iterations: int,
    max_time: float,
    precision: str,
) -> list[str]:
    command = [
        str(driver_path(build_dir, backend)),
        "--matrix", str(matrix),
        "--precision", precision,
        "--index", "int32",
        "--offset", backend.offset,
        "--iterations", str(iterations),
        "--max-time", str(max_time),
        "--output", str(output),
    ]
    # The GPU driver has no thread option: the launch configuration is fixed.
    if backend.device == "cpu":
        command += ["--threads", threads]
    return command


def run_driver(command: Sequence[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    # The driver writes its JSON log itself and will not create the directory.
    output_index = list(command).index("--output")
    Path(command[output_index + 1]).parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        completed = subprocess.run(
            list(command), stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        tail = log.read_text().strip().splitlines()[-15:]
        raise SystemExit(
            "driver failed: " + " ".join(command) + "\n" + "\n".join(tail)
        )


def load_runs(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)["runs"]


def stage_verify(
    build_dir: Path,
    backends: Sequence[Backend],
    specs: Sequence[MatrixSpec],
    data_root: Path,
    work_dir: Path,
    precision: str,
) -> None:
    """Every backend must agree with Eigen on every matrix before anything is timed."""
    print(f"[verify] checking {len(backends)} backends on {len(specs)} matrices "
          f"(tolerance {VERIFICATION_TOLERANCE:g})")
    failures: list[str] = []
    for spec in specs:
        row = [f"  {spec.name:<14}"]
        for backend in backends:
            output = work_dir / "verify" / f"{spec.name}_{backend.key}.json"
            command = driver_command(
                build_dir, backend, matrix_path(data_root, spec), output,
                # A handful of iterations: this stage checks the answer, not the speed.
                threads="1", iterations=3, max_time=30.0, precision=precision,
            )
            run_driver(command, work_dir / "logs" / f"verify_{spec.name}_{backend.key}.log")
            errors = [run["verification_error"] for run in load_runs(output)]
            worst = max(errors) if errors else float("nan")
            ok = errors and worst == worst and worst <= VERIFICATION_TOLERANCE
            row.append(f"{backend.label} {worst:.1e} {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{backend.label} on {spec.name}: max error {worst:.3e}")
        print("  ".join(row), flush=True)

    if failures:
        raise SystemExit(
            "[verify] refusing to benchmark an incorrect kernel:\n  "
            + "\n  ".join(failures)
        )
    print("[verify] every backend matches the Eigen reference")


def stage_benchmark(
    build_dir: Path,
    backends: Sequence[Backend],
    specs: Sequence[MatrixSpec],
    data_root: Path,
    output_dir: Path,
    *,
    threads: str,
    iterations: int,
    max_time: float,
    precision: str,
) -> None:
    print(f"[benchmark] {len(backends)} backends x {len(specs)} matrices, "
          f"threads={threads}, <= {iterations} iterations or {max_time}s each")
    for spec in specs:
        for backend in backends:
            output = output_dir / "raw" / f"{spec.name}_{backend.key}.json"
            command = driver_command(
                build_dir, backend, matrix_path(data_root, spec), output,
                threads=threads, iterations=iterations, max_time=max_time,
                precision=precision,
            )
            started = time.time()
            run_driver(command, output_dir / "logs" / f"{spec.name}_{backend.key}.log")
            print(f"  {spec.name:<14} {backend.label:<10} {time.time() - started:6.1f}s",
                  flush=True)


# --------------------------------------------------------------------------
# Slurm
# --------------------------------------------------------------------------


def on_login_node() -> bool:
    """sbatch exists and we are not inside an allocation."""
    return shutil.which("sbatch") is not None and "SLURM_JOB_ID" not in os.environ


SBATCH_TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={walltime}
#SBATCH --output={output_dir}/slurm-%x-%j.out
{extra_directives}
set -euo pipefail

cd {workdir}
{modules}
# Re-enter this pipeline inside the allocation, where the executor is local by
# construction: the stages that need the hardware run here, the figure is drawn
# afterwards on whichever machine collects the results.
{python} {script} \\
  --stage verify,benchmark \\
  --executor local \\
  --build-dir "{build_dir}" \\
  --data-root "{data_root}" \\
  --output-dir "{output_dir}" \\
  --backends "{backends}" \\
  --threads "{threads}" \\
  --iterations {iterations} \\
  --max-time {max_time} \\
  --precision {precision} \\
  --no-download
"""


def submit_slurm(args: argparse.Namespace, backends: Sequence[Backend]) -> None:
    directives = []
    if args.slurm_account:
        directives.append(f"#SBATCH --account={args.slurm_account}")
    if args.slurm_partition:
        directives.append(f"#SBATCH --partition={args.slurm_partition}")
    if args.slurm_mem:
        directives.append(f"#SBATCH --mem={args.slurm_mem}")
    # Only ask for a GPU if a GPU backend is actually in the run.
    if any(backend.device == "gpu" for backend in backends) and args.slurm_gpus:
        directives.append(f"#SBATCH --gres=gpu:{args.slurm_gpus}")

    modules = "\n".join(f"module load {name}" for name in args.slurm_module)

    script = SBATCH_TEMPLATE.format(
        job_name=args.slurm_job_name,
        cpus=args.slurm_cpus,
        walltime=args.slurm_time,
        extra_directives="\n".join(directives),
        modules=modules,
        workdir=Path.cwd(),
        python=sys.executable,
        script=Path(__file__).resolve(),
        build_dir=args.build_dir,
        data_root=args.data_root,
        output_dir=args.output_dir,
        backends=",".join(backend.key for backend in backends),
        threads=args.threads,
        iterations=args.iterations,
        max_time=args.max_time,
        precision=args.precision,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_path = args.output_dir / "spmv_pipeline.sbatch"
    script_path.write_text(script)
    print(f"[slurm] wrote {script_path}")

    if args.dry_run:
        print("[slurm] --dry-run: not submitting")
        return

    completed = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, check=True
    )
    print(f"[slurm] {completed.stdout.strip()}")
    job_id = completed.stdout.strip().split()[-1]
    if args.no_wait:
        print(f"[slurm] not waiting; when job {job_id} finishes, rerun with --stage figure")
        return

    print(f"[slurm] waiting for job {job_id}", flush=True)
    while True:
        state = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        if not state:
            break
        print(f"  job {job_id}: {state}", flush=True)
        time.sleep(args.poll_seconds)
    print(f"[slurm] job {job_id} left the queue")


# --------------------------------------------------------------------------
# Stage 4: the figure
# --------------------------------------------------------------------------

#: Categorical slots 1-3 of the validated palette, assigned in fixed order and
#: never cycled. Validated with the skill's checker for three slots, all pairs,
#: light surface: worst CVD dE 9.2, worst normal-vision dE 24.0.
SERIES_COLORS = {"SPCraft": "#2a78d6", "oneMKL": "#eb6834", "cuSPARSE": "#1baf7a"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#84837c"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"


def collect_results(output_dir: Path, specs: Sequence[MatrixSpec]) -> list[dict]:
    """Best configuration per (matrix, backend), by median time."""
    rows: list[dict] = []
    for spec in specs:
        for backend in BACKENDS:
            path = output_dir / "raw" / f"{spec.name}_{backend.key}.json"
            if not path.is_file():
                continue
            runs = load_runs(path)
            if not runs:
                continue
            # A CPU driver sweeps thread counts; the headline is its best.
            best = min(runs, key=lambda run: run["median_seconds"])
            rows.append(
                {
                    "matrix": spec.name,
                    "category": spec.category,
                    "backend": backend.label,
                    "device": backend.device,
                    "rows": best["rows"],
                    "nonzeros": best["nonzeros"],
                    "threads": best.get("threads", 0),
                    "median_ms": best["median_seconds"] * 1e3,
                    "gflops": best["gflops"],
                    "gbytes_per_second": best["gbytes_per_second"],
                    "verification_error": best["verification_error"],
                }
            )
    return rows


def write_table(rows: Sequence[dict], output_dir: Path) -> Path:
    """The table view: the relief the palette's contrast WARN obliges, and the
    numbers behind every bar."""
    path = output_dir / "spmv_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "matrix", "category", "backend", "device", "rows", "nonzeros",
        "threads", "median_ms", "gflops", "gbytes_per_second", "verification_error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    return path


def stage_figure(output_dir: Path, figure_dir: Path, specs: Sequence[MatrixSpec],
                 precision: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = collect_results(output_dir, specs)
    if not rows:
        raise SystemExit(f"[figure] no results under {output_dir / 'raw'}; run --stage benchmark")

    figure_dir.mkdir(parents=True, exist_ok=True)
    table_path = write_table(rows, figure_dir)
    print(f"[figure] wrote {table_path}")

    labels = [b.label for b in BACKENDS if any(r["backend"] == b.label for r in rows)]
    # Order the matrices by nonzero count: the x axis then reads as problem size.
    order = sorted({r["matrix"] for r in rows},
                   key=lambda name: next(r["nonzeros"] for r in rows if r["matrix"] == name))
    lookup = {(r["matrix"], r["backend"]): r for r in rows}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.6, 6.6), height_ratios=[1.4, 1.0], constrained_layout=True
    )

    # -- panel A: throughput ------------------------------------------------
    positions = range(len(order))
    span = 0.72
    width = span / len(labels)
    for index, label in enumerate(labels):
        offset = -span / 2 + width * (index + 0.5)
        values = [lookup.get((name, label), {}).get("gflops", 0.0) for name in order]
        # linewidth in the surface colour gives the 2px gap between adjacent bars
        # edgecolor in the surface colour is the prescribed 2px gap between
        # adjacent bars, not a border drawn to separate them.
        top.bar([p + offset for p in positions], values, width * 0.94,
                label=label, color=SERIES_COLORS[label], edgecolor=SURFACE, linewidth=0.8)

    precision_label = {"double": "FP64", "float": "FP32"}[precision]
    top.set_ylabel(f"SpMV throughput   [GFLOP/s, {precision_label}]")
    top.margins(y=0.12)
    top.set_xticks(list(positions))
    top.set_xticklabels([])
    top.grid(axis="y", color=GRID, linewidth=0.6)
    top.set_axisbelow(True)
    for side in ("top", "right"):
        top.spines[side].set_visible(False)
    top.legend(frameon=False, ncol=len(labels), loc="upper left", fontsize=8.5)
    top.set_title("SpMV across a ten-matrix suite: SPCraft, oneMKL and cuSPARSE",
                  loc="left", fontsize=11, fontweight="bold", pad=8)

    # -- panel B: speedup over the CPU baseline -----------------------------
    baseline = "oneMKL" if "oneMKL" in labels else labels[0]
    others = [label for label in labels if label != baseline]
    for index, label in enumerate(others):
        offset = -0.36 + (0.72 / max(len(others), 1)) * (index + 0.5)
        speedups = []
        for name in order:
            reference = lookup.get((name, baseline), {}).get("median_ms")
            mine = lookup.get((name, label), {}).get("median_ms")
            speedups.append(reference / mine if reference and mine else 0.0)
        bar_width = (0.72 / max(len(others), 1)) * 0.94
        bottom.bar([p + offset for p in positions], speedups, bar_width,
                   color=SERIES_COLORS[label], edgecolor=SURFACE, linewidth=0.8, label=label)
        # Direct-label the extremes only: a number on every bar goes unread, and
        # the CSV beside this figure carries every value.
        present = [(p, v) for p, v in zip(positions, speedups) if v > 0]
        if present:
            for position, value in (max(present, key=lambda pv: pv[1]),
                                    min(present, key=lambda pv: pv[1])):
                bottom.annotate(f"{value:.1f}x", (position + offset, value),
                                textcoords="offset points", xytext=(0, 3), ha="center",
                                fontsize=7.0, color=INK_SECONDARY)

    bottom.axhline(1.0, color=INK_MUTED, linewidth=1.0)
    # Headroom so the tallest bar's direct label is not clipped by the frame.
    bottom.margins(y=0.16)
    bottom.set_ylabel(f"Speedup over {baseline}")
    bottom.set_xticks(list(positions))
    bottom.set_xticklabels(
        [f"{name}\n{lookup[(name, labels[0])]['nonzeros'] / 1e6:.1f}M nnz"
         if (name, labels[0]) in lookup else name for name in order],
        fontsize=7.6, color=INK_SECONDARY, rotation=30, ha="right",
        rotation_mode="anchor",
    )
    bottom.grid(axis="y", color=GRID, linewidth=0.6)
    bottom.set_axisbelow(True)
    for side in ("top", "right"):
        bottom.spines[side].set_visible(False)
    bottom.legend(frameon=False, ncol=len(others), loc="upper left", fontsize=8.5)

    cpu_threads = next((r["threads"] for r in rows if r["device"] == "cpu"), 0)
    figure.text(
        0.005, -0.030,
        f"Median of the timed iterations. CPU backends at their best thread count "
        f"(up to {cpu_threads}); cuSPARSE on one GPU. Every result verified against "
        f"an Eigen reference before timing. Matrices ordered by nonzero count.",
        fontsize=6.8, color=INK_MUTED, ha="left",
    )

    for suffix in ("png", "pdf"):
        path = figure_dir / f"spmv_comparison.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[figure] wrote {path}")
    plt.close(figure)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", default="all",
                        help="comma-separated: dataset, verify, benchmark, figure, or all")
    parser.add_argument("--executor", choices=("auto", "local", "slurm"), default="auto",
                        help="auto submits to Slurm on a login node, else runs here")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset/suitsparse"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmarks/results/spmv_pipeline"))
    parser.add_argument("--figure-dir", type=Path, default=Path("benchmarks/plots/figures"),
                        help="where the figure and its table are written")
    parser.add_argument("--backends", default="spcraft,mkl,cusparse")
    parser.add_argument("--matrices", default="",
                        help="comma-separated subset; default is all ten")
    parser.add_argument("--threads", default="",
                        help="thread counts for the CPU drivers; empty means powers of two")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--max-time", type=float, default=10.0)
    parser.add_argument("--precision", default="double", choices=("float", "double"))
    parser.add_argument("--no-download", action="store_true")

    slurm = parser.add_argument_group("slurm")
    slurm.add_argument("--slurm-account", default=os.environ.get("SPCRAFT_SLURM_ACCOUNT", ""))
    slurm.add_argument("--slurm-partition", default=os.environ.get("SPCRAFT_SLURM_PARTITION", ""))
    slurm.add_argument("--slurm-cpus", type=int, default=16)
    slurm.add_argument("--slurm-gpus", type=int, default=1)
    slurm.add_argument("--slurm-mem", default="64G")
    slurm.add_argument("--slurm-time", default="02:00:00")
    slurm.add_argument("--slurm-job-name", default="spcraft-spmv")
    slurm.add_argument("--slurm-module", action="append", default=[],
                       help="module load ... inside the job; repeatable")
    slurm.add_argument("--poll-seconds", type=int, default=30)
    slurm.add_argument("--no-wait", action="store_true")
    slurm.add_argument("--dry-run", action="store_true",
                       help="write the sbatch script without submitting it")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    stages = {s.strip() for s in args.stage.split(",") if s.strip()}
    if "all" in stages:
        stages = {"dataset", "verify", "benchmark", "figure"}

    names = {n.strip() for n in args.matrices.split(",") if n.strip()}
    specs = [s for s in MATRICES if not names or s.name in names]
    if names - {s.name for s in specs}:
        raise SystemExit(f"unknown matrices: {sorted(names - {s.name for s in specs})}")
    wanted = {b.strip() for b in args.backends.split(",") if b.strip()}

    if "dataset" in stages:
        summary = stage_dataset(args.data_root, specs, allow_download=not args.no_download)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "dataset.json").write_text(json.dumps(summary, indent=2))

    needs_hardware = stages & {"verify", "benchmark"}
    use_slurm = args.executor == "slurm" or (args.executor == "auto" and on_login_node())

    if needs_hardware and use_slurm:
        backends = [b for b in BACKENDS if b.key in wanted]
        print("[executor] Slurm login node detected"
              if args.executor == "auto" else "[executor] Slurm requested")
        submit_slurm(args, backends)
    elif needs_hardware:
        print(f"[executor] running locally on {os.uname().nodename}")
        backends = available_backends(args.build_dir, wanted)
        if not backends:
            raise SystemExit(f"no requested driver was built under {args.build_dir}")
        if "verify" in stages:
            stage_verify(args.build_dir, backends, specs, args.data_root,
                         args.output_dir, args.precision)
        if "benchmark" in stages:
            stage_benchmark(args.build_dir, backends, specs, args.data_root, args.output_dir,
                            threads=args.threads, iterations=args.iterations,
                            max_time=args.max_time, precision=args.precision)

    if "figure" in stages:
        stage_figure(args.output_dir, args.figure_dir, specs, args.precision)


if __name__ == "__main__":
    main()
