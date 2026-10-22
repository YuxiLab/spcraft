# Sparse direct solver baselines

This standalone harness compares double-precision CPU LU in SuperLU_MT and
PaStiX. The downloaded upstream checkouts are under
`refernce/sparse-direct/{superlu_mt,pastix}` and retain their own licenses.
PaStiX includes its pinned SpM and morse_cmake submodules.

The initial lair deployment uses a private directory under the remote SpCraft
checkout, `refernce/sparse-direct-20261022`. It contains `src/`, `harness/`,
`packages/`, `deps/`, `install/`, `build/`, `logs/`, and `results/`.
Nothing is installed system-wide. `harness/provenance.json` records source
commits, submodule commits, and the cxxopts header checksum.

Compact samples, summaries, matrix hashes, and source/environment provenance
from the completed run are checked in under `baseline_20261022/`. Full logs and
generated matrix files remain in the ignored result directory.

## Workload and measurements

- Full, real, general Matrix Market inputs: 2D Poisson grids of 128 and 256
  points per dimension, and a diagonally dominant, nonsymmetric 24-cubed
  convection-diffusion stencil.
- Identical input matrices and the known solution `x = 1` for both solvers.
- Thread counts 1, 2, 4, 8, and 16. One discarded warmup and three measured
  fresh processes per solver, input, and thread count.
- Single-threaded OpenBLAS. SuperLU_MT uses OpenMP and MMD on the structure of
  `A^T A`; PaStiX uses its dynamic pthread scheduler and METIS. Both use LU,
  full-rank storage, and one right-hand side, without iterative refinement.
  PaStiX explicitly binds its main thread and workers to distinct cores in the
  allocation; OpenMP uses `OMP_PLACES=cores` and `OMP_PROC_BIND=close`.
- Analysis, factorization, and solve wall times are measured separately.
  Their sum excludes input loading, runtime initialization, correctness checks,
  and cleanup; it is not end-to-end process wall time.
- Every run must have infinity-norm backward error at most `1e-10` and
  infinity-norm forward error at most `1e-7`. Failed runs stop the suite.
- Raw logs, individual samples, matrix hashes, medians, speedups, CPU details,
  dependency hashes, and dynamic library linkage are retained in `results/`.

These are initial solver-configuration baselines. Different orderings and pivot
strategies affect fill and runtime: SuperLU_MT uses partial pivoting, while
PaStiX uses static pivoting. Factor-entry counts also use different internal
storage accounting. Do not interpret these runs as an isolated comparison of
schedulers or assume these synthetic cases represent arbitrary sparse matrices.

## Build and run on lair

All builds and workloads run through Slurm on compute nodes. The scripts request
16 CPU cores and 32 GB in `general` with account `yuxilab` and QOS `normal`,
constrained to the `cpujobs` nodes. They do not request GPUs. Use a different
account, partition, or constraint when the actual server configuration requires
it. The build script checks that it is running on a lair compute node.

Stage the source trees under `src/`, this harness under
`harness/benchmarks/{direct,superlu_mt,pastix}`, and a cxxopts header under
`harness/dependencies/include/`. Supply source provenance as
`harness/provenance.json`. Source archives must exclude `.git`, AppleDouble
`._*` files, and macOS extended attributes.

The Ubuntu 22.04 deployment extracts downloaded `liblapacke-dev`, `liblapacke`,
`libtmglib3`, `libmetis-dev`, and `libmetis5` packages from `packages/` into the
private `deps/` directory. It uses the server's existing OpenBLAS, hwloc,
GCC/G++, CMake, and Python. No root package installation is needed.

From the staged baseline root on lair, submit:

```bash
sbatch harness/benchmarks/direct/run_lair.sbatch "$PWD"
```

Inspect the job's status and log before using `results/summary.csv` or
`results/summary.md`. Re-running the suite in the same directory replaces that
directory's result files; use a fresh root to preserve an earlier experiment.
