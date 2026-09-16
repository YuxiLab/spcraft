# SpMV scaling and profiling report

## Method

- Thread sweep: 1, 2, 4, 8, 16 with close core binding.
- FP64 CSR with 32-bit column indices and row offsets; every backend result is checked against Eigen before timing.
- Latency is the median of warmed-up iterations. Process CPU work is the mean because Linux can batch live-worker accounting updates.
- SPCraft worker wall/CPU clocks are collected in a separate instrumented stage, so profiler overhead does not enter the primary latency.
- ER and R-MAT are deterministic generated graphs; webbase-1M is loaded from `dataset/output/`.

## 16-thread comparison

| Dataset | SPCraft ms | oneMKL ms | SPCraft speedup | oneMKL speedup | SPCraft CPU ms | oneMKL CPU ms | wall imbalance | NNZ imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ER_V50000_D16 | 0.0625 | 0.0626 | 13.59× | 13.85× | 2.2573 | 1.5061 | 1.059× | 1.008× |
| RMAT_S16_E16 | 0.5422 | 0.1027 | 3.97× | 19.25× | 9.7571 | 1.8475 | 4.642× | 4.861× |
| webbase-1M | 0.7267 | 0.3145 | 7.34× | 17.62× | 15.0468 | 6.1652 | 2.795× | 2.773× |

## Interpretation

- **ER_V50000_D16:** SPCraft scales 13.59× and oneMKL 13.85×. At 16 threads oneMKL is 1.00× faster; SPCraft's static row assignment is balanced.
- **RMAT_S16_E16:** SPCraft scales 3.97× and oneMKL 19.25×. At 16 threads oneMKL is 5.28× faster; worker time follows the unequal nonzero assignment.
- **webbase-1M:** SPCraft scales 7.34× and oneMKL 17.62×. At 16 threads oneMKL is 2.31× faster; worker time follows the unequal nonzero assignment.

MKL worker-level bars are intentionally absent: oneMKL's sparse inspector API does not expose its internal worker intervals. Its process CPU work remains in the table and figure annotations.

## Cache counters

The counter experiment targets webbase-1M at 16 threads and requests generic cache, L1-data, and LLC load/miss events through `perf stat`.

- SPCraft: unavailable (Access to performance monitoring and observability operations is limited.). No cache-miss rate is inferred from timing data.
- oneMKL: unavailable (Access to performance monitoring and observability operations is limited.). No cache-miss rate is inferred from timing data.
