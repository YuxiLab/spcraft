# Lair sparse LU baseline — 2026-10-22

Both libraries were downloaded locally and built on Slurm compute nodes. The final run was job `129327` on `lair-g3`, which completed with exit code 0 in 75 seconds. No solver builds or benchmark workloads ran on the login node.

The allocation requested 16 physical CPU cores with SMT excluded for execution and 32 GiB of memory. Slurm accounts for 32 hardware CPU threads in this allocation. The node has AMD EPYC 9354 processors and was shared with other jobs.

## Source revisions

- superlu_mt: `f2e50bfcbb7b20ef304c84cf92c038c283019863` (https://github.com/xiaoyeli/superlu_mt.git).
- pastix: `ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2` (https://gitlab.inria.fr/solverstack/pastix.git).

PaStiX submodule commits and the cxxopts header checksum are in the retained provenance manifest. Both builds used GCC, release optimization, the same system OpenBLAS with one BLAS thread, and 32-bit sparse indices. PaStiX used no MPI, GPU runtime, or low-rank compression.

## Inputs

| Matrix | Dimension | Nonzeros |
|---|---:|---:|
| poisson2d_128 | 16,384 | 81,408 |
| poisson2d_256 | 65,536 | 326,656 |
| convection3d_24 | 13,824 | 93,312 |

Each matrix was supplied as a full general CSC matrix. The RHS was formed from the same input and the known solution `x = 1`. All configurations used LU and one RHS, without iterative refinement.

## Factorization results

Each time below is the median of three measured fresh-process runs after one discarded warmup. The fastest sampled thread count is selected from 1, 2, 4, 8, and 16.

| Matrix | Solver | 1 thread (s) | Best threads | Best time (s) | Speedup | 16 threads (s) |
|---|---|---:|---:|---:|---:|---:|
| poisson2d_128 | superlu_mt | 0.036635 | 2 | 0.024691 | 1.48× | 0.067607 |
| poisson2d_128 | pastix | 0.021292 | 1 | 0.021292 | 1.00× | 0.206018 |
| poisson2d_256 | superlu_mt | 0.212388 | 4 | 0.106569 | 1.99× | 0.259439 |
| poisson2d_256 | pastix | 0.100973 | 1 | 0.100973 | 1.00× | 0.693703 |
| convection3d_24 | superlu_mt | 0.647501 | 4 | 0.214623 | 3.02× | 0.389544 |
| convection3d_24 | pastix | 0.088217 | 1 | 0.088217 | 1.00× | 0.448081 |

Analysis and solve times for every configuration are in [the full summary](../results/direct/20261022-lair/results/summary.md).

## Numerical validation

All 120 runs passed. Maximum normwise infinity-norm backward error was `1.426e-15`; maximum infinity-norm forward error was `4.790e-13`. The acceptance thresholds were `1e-10` and `1e-7`, respectively. PaStiX reported zero static-pivot replacements in these cases.

## Interpretation

- SuperLU_MT uses MMD on the structure of `A^T A`, OpenMP, and partial row pivoting. Its best sampled factorization speedup was 3.02× at four threads on the 3D case.
- PaStiX uses METIS, its dynamic pthread scheduler, and static pivoting. Its serial factorization was faster in these three configurations, but adding workers slowed it down. Explicitly binding its main thread and workers to distinct allocated cores did not resolve the scaling slowdown.
- These measurements do not isolate numerical kernels or schedulers: ordering, pivot strategy, blocking, and storage differ. Small synthetic inputs and a shared node also limit performance conclusions. The result does not establish which library is faster for general sparse workloads.
- Timed analysis, factorization, and solve phases exclude matrix loading, runtime initialization, validation, and cleanup. Phase sums are not process wall times. Library-reported factor-entry counts use different storage accounting.

## Retained artifacts

- [Raw samples](baseline_20261022/samples.json), [CSV summary](baseline_20261022/summary.csv), and [matrix hashes](baseline_20261022/matrices.json).
- [Environment and harness hashes](baseline_20261022/environment.txt), [Slurm completion record](baseline_20261022/slurm-job.txt), and [source provenance](baseline_20261022/provenance.json).
- Default-affinity results from job `129269` are retained separately under `benchmarks/results/direct/20261022-lair/results-default-affinity-job129269/`.
- Remote deployment: `~/workspace/code/iusparse/spcraft/refernce/sparse-direct-20261022` on lair. Local upstream checkouts: `refernce/sparse-direct/`.
- [Harness and rerun instructions](README.md); [source-reading notes](../../docs/direct-solver-references.md).

Compact measured samples, summaries, and provenance are checked in under `baseline_20261022/`. Full raw logs, generated matrix files, and upstream reference checkouts follow the repository’s existing ignore rules.
