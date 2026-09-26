#!/usr/bin/env python3
"""Prepare a Linux/CUDA header environment for native macOS clangd (no build)."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'scripts/clangd'
BASE = ROOT / 'build/clangd-macos'


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def download(item):
    # Names are unique even for GitHub URLs whose last component is just a tag.
    path = BASE / 'downloads' / item['name']
    if not path.exists() or digest(path) != item['sha256']:
        partial = path.with_suffix('.partial')
        run('curl', '--fail', '--location', '--silent', '--show-error',
            '--retry', '3', item['url'], '-o', partial)
        if digest(partial) != item['sha256']:
            raise RuntimeError(f"Checksum mismatch: {item['name']}")
        partial.replace(path)
    return item, path


def install(item, archive):
    stamp = BASE / 'downloads' / (item['name'] + '.installed')
    if stamp.exists() and stamp.read_text() == item['sha256']:
        return
    kind = item['kind']
    destination = BASE / ('sysroot' if kind == 'deb' else
                          'cuda' if kind == 'cuda' else item['name'])
    destination.mkdir(parents=True, exist_ok=True)
    if kind == 'deb':
        members = subprocess.check_output(['ar', 't', str(archive)], text=True)
        member = next(m for m in members.splitlines() if m.startswith('data.tar'))
        data = subprocess.check_output(['ar', 'p', str(archive), member])
        run('tar', '-xf', '-', '-C', destination, input=data)
    else:
        run('tar', '-xf', archive, '--strip-components=1', '-C', destination)
    # Only unpack packages. Never execute Linux binaries or package install scripts.
    stamp.write_text(item['sha256'])
    print(f"Prepared {item['name']}", flush=True)


def database():
    directory = ROOT / 'build'
    relative = lambda path: os.path.relpath(path, directory)
    sysroot = BASE / 'sysroot'
    resource = Path(subprocess.check_output(
        ['clang++', '-print-resource-dir'], text=True).strip())
    flags = ['clang++', '--target=x86_64-linux-gnu',
             f'--sysroot={relative(sysroot)}', '-nostdinc++', '-std=c++20',
             '-DSPCRAFT_USE_CUDA=1', '-I../include', '-I../benchmarks', '-I..',
             f'-I{relative(BASE / "cuda/include")}']
    for dependency in ('fmt/include', 'cxxopts/include', 'boost_preprocessor/include',
                       'fast_matrix_market/include', 'eigen'):
        flags += ['-isystem', relative(BASE / dependency)]
    for path in ('usr/include/c++/12', 'usr/include/x86_64-linux-gnu/c++/12',
                 'usr/include/c++/12/backward', 'usr/include/x86_64-linux-gnu',
                 'usr/include', 'usr/lib/x86_64-linux-gnu/openmpi/include',
                 'usr/lib/llvm-14/lib/clang/14.0.6/include'):
        flags += ['-isystem', relative(sysroot / path)]
    entries = []
    for folder in ('src', 'include', 'tests', 'benchmarks', 'examples'):
        for source in sorted((ROOT / folder).rglob('*')):
            if source.suffix not in ('.cu', '.cuh', '.cpp', '.hpp', '.h'):
                continue
            # Per AGENTS.md, .cuh files contain host declarations, not kernels.
            cuda = source.suffix == '.cu'
            command = flags + ['-x', 'cuda' if cuda else 'c++']
            if cuda:
                # Wrappers must precede explicit GCC include paths for device new/delete.
                command[1:1] = ['-isystem', relative(resource / 'include/cuda_wrappers')]
                command += [f'--cuda-path={relative(BASE / "cuda")}', '--cuda-host-only',
                            '-nocudalib', '--cuda-gpu-arch=sm_80', '-Wno-invalid-constexpr']
            else:
                # Apple Clang accepts the frontend option, but not the driver -fopenmp.
                command += ['-Xclang', '-fopenmp']
            entries.append(dict(directory=str(directory), file=str(source),
                                arguments=command + ['-c', relative(source)]))
    path = BASE / 'compile_commands.json'
    path.write_text(json.dumps(entries, indent=2) + '\n')
    active = directory / 'compile_commands.json'
    if not active.is_symlink():
        active.symlink_to('clangd-macos/compile_commands.json')
    print(f'Activated {len(entries)} parsing commands in {active}')


def check(clangd):
    sources = ('src/cuda/cuSpMV.cu', 'src/cuda/cuPageRank.cu', 'src/cuda/cuda_error.cu',
               'tests/test_cucsr_matrix.cu', 'tests/test_cuspmv.cu', 'tests/test_cupagerank.cu',
               'benchmarks/cusparse/SpMVBenchmark.cu', 'include/core/cuCsrMatrix.cuh',
               'include/kernel/cuSpMV.cuh', 'include/SpCraft.h', 'tests/test_mtspmv.cpp',
               'src/utils/MatrixInput.cpp')
    failed = []
    for source in sources:
        log = BASE / (Path(source).stem + '.check.log')
        with log.open('w') as output:
            # Parse the whole file; only limit per-token editor/refactoring tests.
            # Apple clangd's ExtractFunction self-test rejects valid break/continue.
            result = subprocess.run([clangd, f'--check={ROOT / source}', '--check-lines=1-1'],
                                    stdout=output, stderr=subprocess.STDOUT)
        summaries = [line for line in log.read_text().splitlines()
                     if 'All checks completed' in line]
        print(f'{source}: {summaries[-1] if summaries else log}', flush=True)
        if result.returncode:
            failed.append(str(log))
    if failed:
        raise RuntimeError('clangd checks failed; see ' + ', '.join(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='also check representative project files')
    parser.add_argument('--clangd', default=shutil.which('clangd'), help='clangd executable to verify')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('This setup is macOS-only; use the normal CMake database on Linux')
    if not shutil.which('clang++') or not args.clangd:
        parser.error('Install Xcode Command Line Tools first: xcode-select --install')
    active = ROOT / 'build/compile_commands.json'
    expected = BASE / 'compile_commands.json'
    if active.exists() or active.is_symlink():
        if not active.is_symlink() or active.resolve() != expected:
            parser.error(f'{active} already exists; preserve/move it before setup')
    (BASE / 'downloads').mkdir(parents=True, exist_ok=True)
    packages = json.loads((SUPPORT / 'macos-deps.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for item, archive in pool.map(download, packages):
            install(item, archive)
    database()
    if args.check:
        check(args.clangd)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
