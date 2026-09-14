#!/usr/bin/env python3
"""Compute structural A*A work in contiguous static OpenMP column blocks."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import mmread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrices', type=Path, nargs='+')
    parser.add_argument('--threads', default='16,128')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.matrices:
        matrix = mmread(path).tocsc()
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.shape[0] != matrix.shape[1]:
            parser.error(f'{path} must be square')
        counts = np.diff(matrix.indptr).astype(np.int64)
        pattern = matrix.copy()
        pattern.data = np.ones(pattern.nnz, dtype=np.int64)
        work = np.asarray(pattern.T @ counts).ravel()
        for threads in map(int, args.threads.split(',')):
            blocks = np.array([chunk.sum() for chunk in np.array_split(work, threads)])
            rows.append(dict(dataset=path.stem, threads=threads, columns=matrix.shape[1],
                             work_total=int(work.sum()), max_column_work=int(work.max()),
                             max_static_block_over_mean=float(blocks.max()/blocks.mean()),
                             min_static_block_over_mean=float(blocks.min()/blocks.mean())))
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
