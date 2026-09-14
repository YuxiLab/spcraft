# SPCraft OpenMP vs oneMKL SpMV performance

## Method

- CPU: 16-core AMD EPYC Milan virtual machine, one hardware thread per core.
- Build: `-O3 -DNDEBUG`, GCC, oneMKL 2026.1 GNU OpenMP LP64 backend.
- Precision: FP64; CSR uses 32-bit row offsets and column indices for both implementations.
- Affinity: `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `OMP_WAIT_POLICY=active`.
- Timing: Google Benchmark real time, median of 5 repetitions, at least 0.15 s per repetition.
- oneMKL CSR handle creation, hints, and `mkl_sparse_optimize` are outside the timed loop.
- Each MKL thread count has a separately optimized handle because its inspector partitions work for the active team size.
- `useful_gb_s` is a modeled useful-byte rate, not a hardware DRAM counter.
- Per-thread imbalance was sampled with 16 pinned OpenMP threads over 20 iterations.

## 8-thread comparison

`MKL/SPCraft` is `SPCraft time / MKL time`: above 1 means MKL is faster.

| Matrix | Rows | NNZ | SPCraft ms | SPCraft GF/s | MKL ms | MKL GF/s | MKL/SPCraft | SPCraft scaling | Time imbalance | NNZ imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ecology1 | 1,000,000 | 4,996,000 | 0.434 | 23.02 | 0.297 | 33.70 | 1.46× | 11.24× | 1.27× | 1.00× |
| roadNet-CA | 1,971,281 | 5,533,214 | 1.676 | 6.60 | 1.881 | 5.88 | 0.89× | 11.54× | 1.03× | 1.10× |
| webbase-1M | 1,000,005 | 3,105,536 | 1.052 | 5.90 | 0.622 | 9.99 | 1.69× | 5.60× | 2.86× | 2.77× |
| amazon0312 | 400,727 | 3,200,440 | 0.518 | 12.36 | 0.643 | 9.96 | 0.81× | 11.30× | 1.17× | 1.18× |
| G3_circuit | 1,585,478 | 7,660,826 | 0.887 | 17.27 | 0.765 | 20.04 | 1.16× | 9.77× | 1.27× | 1.06× |
| thermal2 | 1,228,045 | 8,580,313 | 1.719 | 9.98 | 1.585 | 10.83 | 1.08× | 8.36× | 1.06× | 1.00× |
| cfd2 | 123,440 | 3,087,898 | 0.247 | 25.01 | 0.193 | 32.02 | 1.28× | 10.46× | 1.09× | 1.03× |
| cant | 62,451 | 4,007,383 | 0.367 | 21.84 | 0.372 | 21.56 | 0.99× | 10.12× | 1.05× | 1.01× |
| cop20k_A | 121,192 | 2,624,331 | 0.587 | 8.95 | 0.254 | 20.63 | 2.31× | 4.76× | 1.75× | 2.09× |
| af_shell10 | 1,508,065 | 52,672,325 | 12.960 | 8.13 | 12.361 | 8.52 | 1.05× | 3.56× | 1.50× | 1.00× |

## Aggregate findings

- oneMKL wins on 7/10 matrices at 8 threads.
- Geometric-mean MKL speed relative to SPCraft: **1.21×**.
- Geometric-mean 8-thread scaling: SPCraft **8.09×**, MKL **8.90×**.
- MKL's largest advantage is on `cop20k_A` (2.31×); SPCraft's strongest relative result is on `amazon0312` (1.24× faster than MKL).
- Per-thread time imbalance is `max(thread work time) / mean(thread work time)`. NNZ imbalance uses the same ratio for assigned nonzeros. A value of 1 is perfectly balanced.

## Measurement stability

The 8-thread comparison is the primary result because its median CV is 0.43% and its worst CV is 2.34%. At 16 threads this shared VM is visibly noisier: median CV 1.32%, worst CV 69.96%. The 16-thread entry in the scaling table is useful as a saturation indicator, but small differences there should not be treated as conclusive.

## Geometric-mean scaling across the suite

| Threads | SPCraft speedup | MKL speedup |
|---:|---:|---:|
| 1 | 1.00× | 1.00× |
| 2 | 2.22× | 2.23× |
| 4 | 4.48× | 4.67× |
| 8 | 8.09× | 8.90× |
| 16 | 13.84× | 16.73× |

## Interpretation

SPCraft statically divides rows, while row costs track nonzeros and irregular `x` accesses. Matrices with high NNZ or time imbalance leave early-finishing threads waiting at the implicit OpenMP barrier. Compare each matrix's `*_threads.csv` file to determine whether its limitation is uneven work assignment (high NNZ imbalance) or per-nonzero/cache behavior (time imbalance materially above NNZ imbalance).

- `webbase-1M` and `cop20k_A` expose row-count scheduling imbalance: their NNZ imbalances are 2.77× and 2.09×, closely tracking time imbalances of 2.86× and 1.75×. MKL is 1.69× and 2.31× faster, respectively.
- `amazon0312` and `roadNet-CA` favor SPCraft's lower-overhead kernel: SPCraft is 1.24× and 1.12× faster, respectively.
- `G3_circuit` has only 1.06× NNZ imbalance but 1.27× time imbalance, pointing to irregular vector access or cache locality rather than only unequal nonzero counts.
- `af_shell10` has nearly perfect NNZ balance (1.00×), yet 1.50× time imbalance and only 3.56× SPCraft scaling at 8 threads. Its limit is therefore bandwidth/locality or VM interference, not row-count balance.

The clearest next experiment is an NNZ-balanced static row partition precomputed from `row_ptr`. It retains contiguous row ownership while directly targeting the imbalance seen on `webbase-1M` and `cop20k_A`; a dynamic OpenMP schedule is also worth measuring, but may add scheduling overhead and hurt locality.

The benchmark measures steady-state repeated SpMV with hot allocations. It does not include Matrix Market parsing, CSR construction, or oneMKL inspector setup. Results are specific to this AMD EPYC virtual machine and should not be generalized to Intel CPUs without rerunning the suite.
