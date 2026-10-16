#!/usr/bin/env python3
"""Run correctness-checked serial, OpenMP, and MPI examples in both measurement modes."""

import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess


QUERY = ('SELECT region,mpi.rank,mpi.world.size,profiler.iteration,profiler.repetition,'
         'profiler.warmup,profiler.trace,profiler.wall.ns,time.inclusive.duration.ns '
         'WHERE event.end#region FORMAT json')


def summarize(records, ranks):
    phases = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    coverage = defaultdict(set)
    for row in records:
        if 'profiler.repetition' not in row:
            raise ValueError('Warmup or untagged event leaked into the measurement trace')
        repetition = int(row['profiler.repetition'])
        rank = int(row.get('mpi.rank', 0))
        path = row.get('path', row.get('region'))
        seconds = int(row.get('profiler.wall.ns', row['time.inclusive.duration.ns'])) / 1e9
        phases[path][repetition][rank] += seconds
        if path == 'time_to_solution':
            if rank in coverage[repetition]:
                raise ValueError('Duplicate outer timer for one rank/repetition')
            coverage[repetition].add(rank)
    if not coverage or any(value != set(range(ranks)) for value in coverage.values()):
        raise ValueError('Incomplete outer timer coverage across MPI ranks')
    if set(coverage) != set(range(len(coverage))):
        raise ValueError('Missing repetition identifiers')
    result = {}
    for path, repetitions in phases.items():
        # Several SUMMA stages can execute the same phase within one repetition.
        values = [max(repetitions[rep].get(rank, 0) for rank in range(ranks)) for rep in coverage]
        result[path] = {
            'count': len(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'min': min(values), 'max': max(values),
            'stddev': statistics.stdev(values) if len(values) > 1 else 0,
        }
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('build/profiler'))
    parser.add_argument('--caliper-prefix', type=Path, default=Path('build/profiler-caliper-install'))
    parser.add_argument('--matrix', type=Path, default=Path('examples/spcraft/inputs/summa_2d.mtx'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--launches', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=1000)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--mpiexec', default='mpiexec')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if args.launches < 1 or args.repetitions < 1 or args.warmup < 0:
        parser.error('launches/repetitions must be positive; warmup must be nonnegative')
    output = args.output or Path('build/comment-profiler-evaluation') / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output = (root / output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = json.loads((root / '.profiler/regions.json').read_text())
    cases = [
        ('serial-spmv', 'ex01_spmv', 1, 1),
        ('openmp-spmv', 'ex01_spmv', 1, 2),
        ('serial-spgemm', 'ex02_spgemm', 1, 1),
        ('openmp-spgemm', 'ex02_spgemm', 1, 2),
        ('mpi-summa-1', 'e08_summa_2d', 1, 1),
        ('mpi-summa-4', 'e08_summa_2d', 4, 1),
    ]
    runs = [(case, mode, launch) for case in cases for mode in ('benchmark', 'profile')
            for launch in range(args.launches)]
    random.Random(2026).shuffle(runs)
    summary = {'platform': platform.platform(), 'matrix': str(args.matrix),
               'matrix_sha256': hashlib.sha256((root / args.matrix).read_bytes()).hexdigest(),
               'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'dirty_files': subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).splitlines(),
               'warmups': args.warmup, 'repeat_limit': args.repetitions, 'seconds_limit': args.seconds,
               'order_seed': 2026, 'runs': []}
    query = (root / args.caliper_prefix / 'bin/cali-query').resolve()
    for (name, target, ranks, threads), mode, launch in runs:
        directory = output / f'{name}-{mode}-{launch}'
        directory.mkdir()
        binary = (root / args.build_dir / 'bin' / f'{target}_{mode}').resolve()
        command = [str(binary), '--matrix', str(args.matrix), '--profiler-warmup', str(args.warmup),
                   '--profiler-repeat', str(args.repetitions)]
        if args.seconds is not None:
            command += ['--profiler-seconds', str(args.seconds)]
        mpi = target == 'e08_summa_2d'
        if mpi:
            side = 2 if ranks == 4 else 1
            command += ['--grid-rows', str(side), '--grid-cols', str(side)]
            command = [args.mpiexec, '-n', str(ranks), *command]
        overrides = {
            'OMP_NUM_THREADS': str(threads), 'CALI_CONFIG': '',
            'CALI_SERVICES_ENABLE': 'event,recorder,timer,trace' + (',mpi' if mpi else ''),
            'CALI_TIMER_INCLUSIVE_DURATION': 'true',
            'CALI_RECORDER_FILENAME': str(directory / ('rank-%mpi.rank%.cali' if mpi else 'serial.cali')),
            'CALI_CHANNEL_FLUSH_ON_EXIT': 'true',
        }
        result = subprocess.run(command, cwd=root, env={**os.environ, **overrides},
                                capture_output=True, text=True, timeout=args.timeout)
        (directory / 'run.log').write_text(result.stdout + result.stderr)
        if result.returncode or not ('verification passed' in result.stdout or 'Eigen check passed' in result.stdout):
            raise RuntimeError(f'Correctness/run failure: {directory / "run.log"}')
        traces = sorted(directory.glob('*.cali'))
        if len(traces) != ranks:
            raise ValueError(f'Expected {ranks} traces in {directory}, found {len(traces)}')
        exported = subprocess.run([str(query), '--query', QUERY, *map(str, traces)],
                                  capture_output=True, text=True, check=True)
        records = json.loads(exported.stdout)
        metrics = summarize(records, ranks)
        if args.seconds is None and metrics['time_to_solution']['count'] != args.repetitions:
            raise ValueError('Wrong number of measured repetitions')
        if mode == 'benchmark' and set(metrics) != {'time_to_solution'}:
            raise ValueError('Phase instrumentation leaked into native-library benchmark')
        if mode == 'profile' and len(metrics) < 2:
            raise ValueError('Missing internal phase records')
        run = {'name': f'{name} {mode}', 'server': platform.node(), 'command': command,
               'environment': overrides, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
        (directory / 'profile.json').write_text(json.dumps(
            {'version': 1, 'records': records, 'sources': sources, 'run': run}, indent=2) + '\n')
        summary['runs'].append({'case': name, 'mode': mode, 'launch': launch, 'ranks': ranks,
                                'threads': threads, 'verified': True, 'directory': str(directory),
                                'phases': metrics, 'run': run})
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        overall = metrics['time_to_solution']
        print(f'{name:17} {mode:9} launch={launch} repeats={overall["count"]} '
              f'median={overall["median"]:.6g}s phases={len(metrics)}', flush=True)
    print(f'Artifacts: {output}')


if __name__ == '__main__':
    main()
