#!/usr/bin/env python3
"""Run sequential, correctness-checked SpCraft/CombBLAS comparisons and report noise."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess


def ratio_interval(left, right):
    """Paired bootstrap interval for the ratio of median native call times."""
    rng = random.Random(42)
    ratios = []
    for _ in range(2000):
        indices = rng.choices(range(len(left)), k=len(left))
        ratios.append(statistics.median(left[i] for i in indices)
                      / statistics.median(right[i] for i in indices))
    ratios.sort()
    return ratios[49], ratios[1949]


def summarize(output_dir, plot):
    rows = []
    for path in sorted(output_dir.glob('*.json')):
        runs = json.loads(path.read_text()).get('runs', [])
        if len(runs) % 2:
            raise ValueError(f'incomplete pair in {path}')
        for a,b in zip(runs[::2], runs[1::2]):
            if (a['backend'],b['backend']) != ('SPCraft-Hash-Tuples','CombBLAS-Hash-Tuples'):
                raise ValueError(f'unexpected pair in {path}')
            for key in ('dataset','threads','precision','index_type','offset_type'):
                if a[key] != b[key]:
                    raise ValueError(f'mismatched {key} in {path}')
            if not a['seconds'] or len(a['seconds']) != len(b['seconds']):
                raise ValueError(f'mismatched samples in {path}')
            low,high = ratio_interval(a['seconds'],b['seconds'])
            rows.append(dict(dataset=a['dataset'], types=f"{a['precision']}/{a['index_type']}",
                             threads=a['threads'], samples=len(a['seconds']),
                             spcraft_ms=1000*a['median_seconds'], combblas_ms=1000*b['median_seconds'],
                             ratio=a['median_seconds']/b['median_seconds'], ratio_low=low, ratio_high=high,
                             spcraft_cv_pct=100*a['stddev_seconds']/a['mean_seconds'],
                             combblas_cv_pct=100*b['stddev_seconds']/b['mean_seconds'],
                             max_error=max(a['verification_error'],b['verification_error'])))
    if not rows:
        raise ValueError('no tuple pairs')
    with (output_dir/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    report=['# Paired native tuple comparison', '',
            'Both kernels read identical DCSC inputs and return column/row-sorted '
            'COO arrays (SpCraft TupleEntry; CombBLAS std::tuple). Timings include allocation, '
            'symbolic, numeric, sorting and destruction. No output conversion occurs. '
            'Full independent Eigen checks precede timing at each thread count. '
            'Only these two kernels run during warmup and timing. Three warmup rounds '
            'precede measured pairs, whose order alternates AB/BA.', '',
            'Ratio = SpCraft / CombBLAS median. Below one favors SpCraft. Intervals '
            'are paired bootstrap 95% intervals over rounds, not uncertainty across '
            'node allocations. CV describes sample variability separately from the gap.', '',
            '| Matrix | Threads | Pairs | SpCraft ms | CombBLAS ms | Ratio [95% CI] | CV % (Sp / Comb) |',
            '|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        report.append(f"| {r['dataset']} | {r['threads']} | {r['samples']} | {r['spcraft_ms']:.3f} "
                      f"| {r['combblas_ms']:.3f} | {r['ratio']:.3f} [{r['ratio_low']:.3f}, {r['ratio_high']:.3f}] "
                      f"| {r['spcraft_cv_pct']:.1f} / {r['combblas_cv_pct']:.1f} |")
    report += ['', f"Maximum absolute verification error: {max(r['max_error'] for r in rows):.6g}."]
    (output_dir/'report.md').write_text('\n'.join(report)+'\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        for ds in sorted({r['dataset'] for r in rows}):
            series=sorted((r for r in rows if r['dataset']==ds),key=lambda r:r['threads'])
            x=[r['threads'] for r in series]
            line,=ax.plot(x,[r['ratio'] for r in series],marker='o',label=ds)
            ax.fill_between(x,[r['ratio_low'] for r in series],[r['ratio_high'] for r in series],
                            color=line.get_color(),alpha=.1)
        ax.axhline(1,color='black',linestyle='--',linewidth=1)
        ax.set(xlabel='OpenMP threads',ylabel='SpCraft / CombBLAS median time',
               title='Paired DCSC-to-native-tuples hash SpGEMM')
        ax.set_xticks(sorted({r['threads'] for r in rows}));ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.savefig(output_dir/'comparison.png',dpi=180);fig.savefig(output_dir/'comparison.pdf')
        plt.close(fig)
    print((output_dir/'report.md').read_text(),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--threads", default="1,4,8,16")
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--max-time", type=float, default=40)
    parser.add_argument("--cases", default="er8,er32,rmat,amazon0312,cant,ecology1")
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="write tables without Matplotlib")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyse_only:
        summarize(args.output_dir, plot=not args.no_plot)
        return
    cases = {
        "er8": ["--vertices", "20000", "--degree", "8"],
        "er32": ["--vertices", "20000", "--degree", "32"],
        "rmat": ["--generator", "rmat", "--scale", "12", "--edge-factor", "8"],
        "hypersparse": ["--vertices", "20000", "--degree", "8", "--column-stride", "64"],
    }
    for name in ("amazon0312", "cant", "ecology1"):
        cases[name] = ["--matrix", str(args.dataset_dir / "suitsparse" / name / f"{name}.mtx")]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "false",
           "OMP_PROC_BIND": "close", "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "active"}
    metadata = {"platform": platform.platform(), "hostname": platform.node(),
                "slurm": {k: os.environ[k] for k in
                          ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")
                          if k in os.environ},
                "binary": str(args.binary),
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "environment": {k: v for k, v in env.items() if k.startswith("OMP_")},
                "lscpu": subprocess.check_output(["lscpu"], text=True), "commands": []}
    for name in args.cases.split(","):
        if name not in cases:
            parser.error(f"unknown case: {name}")
        command = [str(args.binary), *cases[name], "--threads", args.threads,
                   "--iterations", str(args.iterations), "--max-time", str(args.max_time),
                   "--precision", "double", "--index", "int32", "--offset", "int32",
                   "--seed", "42", "--output", str(args.output_dir / f"{name}.json")]
        print("Running:", " ".join(command), flush=True)
        metadata["commands"].append(command)
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (args.output_dir / f"{name}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print((args.output_dir / f"{name}.log").read_text(), flush=True)
    summarize(args.output_dir, plot=not args.no_plot)


if __name__ == "__main__":
    main()
