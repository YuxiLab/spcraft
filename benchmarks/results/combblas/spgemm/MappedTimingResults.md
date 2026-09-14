# Mapped SpGEMM timing on BigRed

The source-mapped baseline is within 5% of CombBLAS in 37 of 42 measured configurations. At 128 threads, five of seven matrices are within 2.4%; ER degree 32 has a noisy 6.1% median gap, and ecology1 is 11.9% slower in SpCraft.

This run measures `spgemm_combblas_openmp` against the unmodified `LocalSpGEMMHash(..., false, false, true)` at commit `2381a1f7a028b9f56d8531c0b7b157ae2ab27572`. Earlier general-kernel timing reports do not measure this mapped implementation. The [source mapping](../../../combblas/SourceMapping.md) lists the retained arithmetic, validation and ownership adaptations.

## Run and correctness

- BigRed CPU debug job `8170854`, exclusive `nid0639`, completed with exit code 0 in 8 minutes 1 second. Two AMD EPYC 7742 sockets, 128 physical cores, eight NUMA domains; one MPI rank.
- GCC 14.2.0 through Cray CC, Release, `-O3 -DNDEBUG -std=c++20 -fPIE -fopenmp`. Both kernels are instantiated in the same benchmark translation unit. FP64 values and int32 indices/offsets.
- Threads: 1, 4, 16, 32, 64, 128. `OMP_DYNAMIC=false`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `OMP_WAIT_POLICY=active`; serial input preparation.
- Three warmup rounds, then alternating AB/BA pairs. Up to 20 pairs or 45 seconds per configuration: 830 measured pairs total. One-thread ER degree 32, Amazon and cant have 17, 19 and 14 pairs respectively; all other configurations have 20.
- Actual upstream helper/kernel comparisons, tuple ownership, DCSC container and column-kernel tests passed before timing. The source-mapping test was compiled with `SPCRAFT_TEST_COMBBLAS=1` and covers 16 canonical products at 1, 2 and 4 threads.
- All 84 timed-backend records passed full Eigen structure/value verification before measurement at each thread count; maximum absolute verification error is zero. Recorded source hashes match the local implementation and pinned reference.

Both paths read identical canonical DCSC A and B and return sorted `std::tuple<int,int,double>` arrays. The timer includes lookup/setup, work estimation, symbolic counting, prefixes, numeric accumulation, sorting, allocation and destruction. Input conversion and Eigen verification are excluded. Neither timed path converts its output.

## 128-thread medians

Ratio is SpCraft time / CombBLAS time; below one favors SpCraft. Intervals are paired bootstrap 95% intervals from 2,000 resamples. CV is within-run sample variability.

| Matrix | SpCraft ms | CombBLAS ms | Ratio [95% CI] | CV % (Sp / Comb) |
|---|---:|---:|---|---|
| ER degree 8 | 4.300 | 4.201 | 1.024 [0.993, 1.034] | 2.61 / 1.94 |
| ER degree 32 | 58.598 | 55.243 | 1.061 [0.837, 1.154] | 8.14 / 8.68 |
| R-MAT | 9.793 | 9.815 | 0.998 [0.989, 1.015] | 1.48 / 2.65 |
| Hypersparse | 4.626 | 4.661 | 0.993 [0.963, 1.009] | 2.13 / 1.02 |
| amazon0312 | 94.459 | 93.578 | 1.009 [0.995, 1.017] | 0.95 / 0.68 |
| cant | 47.876 | 47.519 | 1.008 [0.998, 1.018] | 1.13 / 1.18 |
| ecology1 | 43.713 | 39.066 | 1.119 [1.103, 1.128] | 1.53 / 0.81 |

The ecology1 gap increases from 1.7% at one thread to 8.2% at 16 and 11.9% at 128. Its 128-thread interval excludes parity. This is a remaining performance difference, whose phase-level cause was not measured by this run. The historical general-kernel profiler cannot identify the cause in this mapped baseline.

ER degree 32 at 128 threads has CVs near 8–9% and a ratio interval spanning parity. Its 6.1% median difference is not a precise estimate of a persistent gap. These intervals describe one node allocation and do not quantify variation across allocations.

## Reproduce and inspect

From the prepared BigRed comparison workspace:

```sh
SPCRAFT_NATIVE_ONLY=1 sbatch benchmarks/combblas/slurm/spgemm_columns.sbatch
```

- [All 42 configurations, confidence intervals and CVs](tuples-bigred-8170854/report.md)
- [Machine-readable summary](tuples-bigred-8170854/summary.csv)
- [Thread-scaling plot](tuples-bigred-8170854/comparison.png) and [PDF](tuples-bigred-8170854/comparison.pdf)
- [Runtime metadata and exact commands](tuples-bigred-8170854/metadata.json)
- [Source hashes](tuples-bigred-8170854/source-sha256.txt), [compiler flags](tuples-bigred-8170854/compiler-flags.txt), [test compiler flags](tuples-bigred-8170854/test-compiler-flags.txt)
- [Build, correctness and benchmark log](tuples-bigred-8170854/job-output.txt), [Slurm completion](tuples-bigred-8170854/job-accounting.txt)

Per-matrix JSON files beside the summary preserve every raw sample and the measured kernel identity. The preliminary job `8170809` stopped at the source-revision check before building or timing because the transferred reference lacked Git metadata; restoring metadata and verifying a clean pinned tree preceded the successful run.
