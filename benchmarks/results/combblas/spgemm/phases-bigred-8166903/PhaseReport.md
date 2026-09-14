# SpGEMM phase measurements

Wall phases are measured outside the OpenMP loops. Tables show means so phase contributions add; total-call and diagnostic tables show medians. The sampled worker table is a separate experiment and shows shares of sampled active elapsed time across workers, not fractions of wall time. It excludes scheduling/wait time and may perturb very short columns.

## Whole-call controls and diagnostics

| Input | Threads | CSC control ms | DCSC control ms | Comb native ms | DCSC coarse/control | Comb coarse/control | Dense lookup ms | Static ms | Dense+static ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ER_V20000_D8 | 1 | 75.221 | 82.118 | 83.878 | 1.004 | 1.035 | 74.926 | 81.258 | 75.179 |
| ER_V20000_D8 | 16 | 6.262 | 6.912 | 7.586 | 1.029 | 1.017 | 6.542 | 6.866 | 6.281 |
| ER_V20000_D8 | 128 | 3.620 | 4.294 | 7.157 | 1.133 | 0.981 | 3.853 | 4.360 | 3.751 |
| amazon0312.mtx | 1 | 1084.056 | 1289.117 | 1263.919 | 0.998 | 1.014 | 1092.498 | 1286.911 | 1088.525 |
| amazon0312.mtx | 16 | 88.508 | 106.448 | 248.554 | 0.999 | 1.006 | 89.643 | 265.733 | 222.584 |
| amazon0312.mtx | 128 | 57.799 | 72.072 | 96.477 | 1.013 | 1.012 | 54.768 | 104.777 | 87.308 |
| cant.mtx | 1 | 1850.387 | 1730.669 | 1634.245 | 1.114 | 1.048 | 1759.671 | 1721.794 | 1795.518 |
| cant.mtx | 16 | 134.939 | 127.703 | 132.922 | 1.102 | 1.025 | 129.209 | 128.121 | 132.340 |
| cant.mtx | 128 | 58.369 | 60.665 | 55.062 | 0.996 | 1.019 | 55.020 | 44.152 | 43.961 |
| ecology1.mtx | 1 | 248.090 | 372.209 | 368.951 | 0.983 | 1.039 | 255.917 | 369.738 | 254.839 |
| ecology1.mtx | 16 | 42.391 | 61.837 | 47.872 | 0.996 | 1.001 | 43.593 | 58.959 | 41.681 |
| ecology1.mtx | 128 | 59.337 | 79.015 | 48.365 | 1.005 | 1.028 | 59.048 | 63.893 | 51.451 |

## Wall-clock phases (mean milliseconds)

CombBLAS work and symbolic routines each build additional auxiliary lookup structures internally; their time stays in those routines. The outer lookup row is not the full cost of all CombBLAS lookup operations.

### ER_V20000_D8, 1 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.245 | 0.019 |
| Scratch/thread setup | 0.020 | 0.021 | 0.002 |
| Work bound/capacity | 0.486 | 1.308 | 3.009 |
| Symbolic | 15.793 | 19.078 | 18.120 |
| Prefix/numeric capacity | 0.148 | 0.148 | 0.027 |
| Output allocation/setup | 1.544 | 1.525 | 2.282 |
| Numeric incl. sort/write | 57.519 | 60.170 | 63.372 |
| Cleanup/free | 0.002 | 0.002 | 0.003 |
| Total | 75.512 | 82.497 | 86.834 |
### ER_V20000_D8, 16 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.253 | 0.034 |
| Scratch/thread setup | 0.042 | 0.034 | 0.006 |
| Work bound/capacity | 0.179 | 0.256 | 0.270 |
| Symbolic | 1.082 | 1.251 | 1.268 |
| Prefix/numeric capacity | 0.232 | 0.297 | 0.011 |
| Output allocation/setup | 1.084 | 1.135 | 2.020 |
| Numeric incl. sort/write | 3.650 | 3.813 | 4.079 |
| Cleanup/free | 0.002 | 0.002 | 0.005 |
| Total | 6.271 | 7.041 | 7.692 |
### ER_V20000_D8, 128 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.258 | 0.055 |
| Scratch/thread setup | 0.065 | 0.052 | 0.044 |
| Work bound/capacity | 0.386 | 0.436 | 0.366 |
| Symbolic | 0.272 | 0.311 | 0.464 |
| Prefix/numeric capacity | 0.313 | 0.471 | 0.065 |
| Output allocation/setup | 1.905 | 1.978 | 4.784 |
| Numeric incl. sort/write | 1.083 | 1.041 | 1.220 |
| Cleanup/free | 0.002 | 0.002 | 0.026 |
| Total | 4.026 | 4.549 | 7.024 |
### amazon0312.mtx, 1 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 5.476 | 0.322 |
| Scratch/thread setup | 0.657 | 0.643 | 0.003 |
| Work bound/capacity | 12.618 | 32.350 | 66.709 |
| Symbolic | 235.875 | 295.752 | 279.108 |
| Prefix/numeric capacity | 2.861 | 2.826 | 0.304 |
| Output allocation/setup | 11.687 | 12.240 | 19.935 |
| Numeric incl. sort/write | 820.985 | 937.845 | 914.919 |
| Cleanup/free | 0.004 | 0.003 | 0.005 |
| Total | 1084.687 | 1287.135 | 1281.305 |
### amazon0312.mtx, 16 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 5.470 | 0.339 |
| Scratch/thread setup | 0.801 | 0.710 | 0.011 |
| Work bound/capacity | 4.375 | 6.042 | 9.855 |
| Symbolic | 15.474 | 19.152 | 46.784 |
| Prefix/numeric capacity | 4.283 | 3.752 | 0.142 |
| Output allocation/setup | 11.842 | 12.367 | 20.502 |
| Numeric incl. sort/write | 51.993 | 58.851 | 172.579 |
| Cleanup/free | 0.003 | 0.003 | 0.007 |
| Total | 88.771 | 106.346 | 250.219 |
### amazon0312.mtx, 128 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 5.756 | 0.822 |
| Scratch/thread setup | 1.363 | 1.106 | 0.065 |
| Work bound/capacity | 5.387 | 7.651 | 4.363 |
| Symbolic | 7.032 | 8.847 | 13.920 |
| Prefix/numeric capacity | 6.000 | 6.481 | 0.178 |
| Output allocation/setup | 15.373 | 15.553 | 27.548 |
| Numeric incl. sort/write | 22.763 | 27.037 | 50.736 |
| Cleanup/free | 0.003 | 0.003 | 0.029 |
| Total | 57.922 | 72.435 | 97.661 |
### cant.mtx, 1 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.799 | 0.059 |
| Scratch/thread setup | 0.081 | 0.075 | 0.003 |
| Work bound/capacity | 6.627 | 27.979 | 30.907 |
| Symbolic | 444.227 | 537.897 | 348.063 |
| Prefix/numeric capacity | 0.281 | 0.280 | 0.038 |
| Output allocation/setup | 16.254 | 16.244 | 26.728 |
| Numeric incl. sort/write | 1336.911 | 1345.796 | 1305.226 |
| Cleanup/free | 0.002 | 0.002 | 0.005 |
| Total | 1804.384 | 1929.073 | 1711.029 |
### cant.mtx, 16 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.798 | 0.053 |
| Scratch/thread setup | 0.088 | 0.088 | 0.008 |
| Work bound/capacity | 1.139 | 2.141 | 2.106 |
| Symbolic | 29.006 | 34.793 | 23.389 |
| Prefix/numeric capacity | 0.565 | 0.827 | 0.016 |
| Output allocation/setup | 16.463 | 16.563 | 28.036 |
| Numeric incl. sort/write | 84.502 | 85.467 | 82.760 |
| Cleanup/free | 0.002 | 0.002 | 0.006 |
| Total | 131.766 | 140.679 | 136.375 |
### cant.mtx, 128 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.906 | 0.088 |
| Scratch/thread setup | 0.287 | 0.207 | 0.055 |
| Work bound/capacity | 1.408 | 1.848 | 1.080 |
| Symbolic | 8.078 | 8.348 | 3.958 |
| Prefix/numeric capacity | 0.954 | 1.649 | 0.090 |
| Output allocation/setup | 21.708 | 21.638 | 35.650 |
| Numeric incl. sort/write | 26.479 | 26.746 | 15.373 |
| Cleanup/free | 0.003 | 0.003 | 0.026 |
| Total | 58.917 | 61.346 | 56.320 |
### ecology1.mtx, 1 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 13.922 | 0.778 |
| Scratch/thread setup | 1.274 | 1.307 | 0.004 |
| Work bound/capacity | 10.115 | 39.497 | 46.408 |
| Symbolic | 57.982 | 94.164 | 90.695 |
| Prefix/numeric capacity | 2.395 | 2.384 | 0.522 |
| Output allocation/setup | 10.827 | 12.085 | 18.395 |
| Numeric incl. sort/write | 170.196 | 202.444 | 226.536 |
| Cleanup/free | 0.002 | 0.003 | 0.004 |
| Total | 252.791 | 365.807 | 383.342 |
### ecology1.mtx, 16 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 13.923 | 0.838 |
| Scratch/thread setup | 1.408 | 1.433 | 0.009 |
| Work bound/capacity | 5.088 | 6.046 | 4.001 |
| Symbolic | 4.582 | 6.999 | 7.185 |
| Prefix/numeric capacity | 5.208 | 4.727 | 0.152 |
| Output allocation/setup | 11.028 | 12.279 | 19.696 |
| Numeric incl. sort/write | 15.208 | 16.266 | 16.000 |
| Cleanup/free | 0.004 | 0.003 | 0.007 |
| Total | 42.526 | 61.676 | 47.887 |
### ecology1.mtx, 128 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 15.245 | 1.704 |
| Scratch/thread setup | 4.406 | 2.626 | 0.060 |
| Work bound/capacity | 6.103 | 7.380 | 3.089 |
| Symbolic | 5.627 | 7.174 | 3.083 |
| Prefix/numeric capacity | 12.425 | 11.700 | 0.324 |
| Output allocation/setup | 17.061 | 18.727 | 32.604 |
| Numeric incl. sort/write | 15.010 | 16.567 | 8.805 |
| Cleanup/free | 0.007 | 0.007 | 0.030 |
| Total | 60.640 | 79.427 | 49.699 |

## Sampled numeric worker time

DCSC hashing includes repeated left-column lookup; CombBLAS lookup is timed separately before hashing. Percentages normalize the summed sampled durations. Sampling selects approximately 8,192 evenly spaced candidate columns per call (all columns for smaller inputs); these shares are descriptive, not estimates of a critical thread or an unbiased distribution of irregular work.

| Input | Threads | Backend | Lookup % | Alloc/init % | Hash/expand % | Compact % | Sort % | Write % | Free % | Sampled/coarse total |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ER_V20000_D8 | 1 | DCSC | 0.0 | 3.6 | 34.5 | 9.7 | 46.4 | 3.3 | 2.5 | 1.030 |
| ER_V20000_D8 | 1 | CombBLAS | 6.2 | 4.6 | 26.7 | 9.5 | 44.5 | 6.3 | 2.1 | 1.028 |
| ER_V20000_D8 | 16 | DCSC | 0.0 | 3.5 | 34.6 | 9.8 | 46.3 | 3.6 | 2.3 | 0.985 |
| ER_V20000_D8 | 16 | CombBLAS | 6.0 | 4.6 | 27.2 | 9.7 | 44.8 | 5.9 | 1.9 | 1.036 |
| ER_V20000_D8 | 128 | DCSC | 0.0 | 3.4 | 23.9 | 5.3 | 24.9 | 29.4 | 13.1 | 0.831 |
| ER_V20000_D8 | 128 | CombBLAS | 21.2 | 3.2 | 15.9 | 5.7 | 25.5 | 21.2 | 7.2 | 0.817 |
| amazon0312.mtx | 1 | DCSC | 0.0 | 3.3 | 50.8 | 7.7 | 33.2 | 3.1 | 1.9 | 1.002 |
| amazon0312.mtx | 1 | CombBLAS | 9.7 | 4.4 | 37.6 | 7.7 | 33.3 | 5.4 | 1.8 | 1.003 |
| amazon0312.mtx | 16 | DCSC | 0.0 | 3.2 | 51.3 | 7.5 | 32.5 | 3.4 | 2.0 | 1.001 |
| amazon0312.mtx | 16 | CombBLAS | 9.9 | 4.2 | 38.4 | 7.6 | 32.8 | 5.2 | 1.8 | 1.002 |
| amazon0312.mtx | 128 | DCSC | 0.0 | 2.3 | 75.4 | 2.4 | 10.4 | 6.3 | 3.3 | 0.960 |
| amazon0312.mtx | 128 | CombBLAS | 13.7 | 2.5 | 39.8 | 4.4 | 19.2 | 15.4 | 4.9 | 0.962 |
| cant.mtx | 1 | DCSC | 0.0 | 1.4 | 70.6 | 2.4 | 23.7 | 1.5 | 0.3 | 1.002 |
| cant.mtx | 1 | CombBLAS | 2.8 | 2.1 | 64.3 | 3.0 | 25.3 | 2.2 | 0.3 | 1.002 |
| cant.mtx | 16 | DCSC | 0.0 | 1.4 | 70.8 | 2.4 | 23.8 | 1.3 | 0.3 | 1.001 |
| cant.mtx | 16 | CombBLAS | 2.8 | 2.1 | 64.4 | 3.0 | 25.3 | 2.1 | 0.3 | 0.999 |
| cant.mtx | 128 | DCSC | 0.0 | 1.3 | 74.1 | 1.5 | 10.1 | 10.7 | 2.5 | 1.045 |
| cant.mtx | 128 | CombBLAS | 4.0 | 1.8 | 57.0 | 2.6 | 22.1 | 11.2 | 1.2 | 0.925 |
| ecology1.mtx | 1 | DCSC | 0.0 | 10.6 | 34.6 | 13.0 | 19.9 | 11.4 | 10.5 | 1.006 |
| ecology1.mtx | 1 | CombBLAS | 18.1 | 11.9 | 23.1 | 11.1 | 17.1 | 9.6 | 9.2 | 1.013 |
| ecology1.mtx | 16 | DCSC | 0.0 | 9.9 | 40.5 | 11.6 | 17.4 | 10.5 | 10.1 | 1.004 |
| ecology1.mtx | 16 | CombBLAS | 19.0 | 11.7 | 22.9 | 10.7 | 16.9 | 9.7 | 9.1 | 0.998 |
| ecology1.mtx | 128 | DCSC | 0.0 | 0.5 | 81.6 | 0.4 | 1.2 | 10.8 | 5.5 | 0.991 |
| ecology1.mtx | 128 | CombBLAS | 26.7 | 19.5 | 17.3 | 4.8 | 14.3 | 7.4 | 10.0 | 0.855 |

## Tuple-to-DCSC conversion (mean milliseconds)

| Input | Threads | Native call | Compression | DCSC free | Tuple free |
|---|---:|---:|---:|---:|---:|
| ER_V20000_D8 | 1 | 83.756 | 7.076 | 0.003 | 0.001 |
| ER_V20000_D8 | 16 | 7.199 | 6.567 | 0.002 | 0.001 |
| ER_V20000_D8 | 128 | 5.444 | 7.010 | 0.002 | 0.001 |
| amazon0312.mtx | 1 | 1263.950 | 53.336 | 0.004 | 0.001 |
| amazon0312.mtx | 16 | 248.334 | 54.672 | 0.003 | 0.001 |
| amazon0312.mtx | 128 | 94.941 | 59.815 | 0.003 | 0.001 |
| cant.mtx | 1 | 1633.053 | 70.836 | 0.003 | 0.001 |
| cant.mtx | 16 | 132.534 | 73.145 | 0.003 | 0.001 |
| cant.mtx | 128 | 52.776 | 78.866 | 0.003 | 0.001 |
| ecology1.mtx | 1 | 369.330 | 46.545 | 0.003 | 0.001 |
| ecology1.mtx | 16 | 47.821 | 48.658 | 0.003 | 0.001 |
| ecology1.mtx | 128 | 43.119 | 76.159 | 0.003 | 0.001 |

## Interpretation limits

- Dense lookup is valid only when sorted unique DCSC column IDs cover every logical column (nzc==n). Otherwise it falls back to the original lookup.
- Static and dense+static change scheduling only, or scheduling plus that lookup shortcut. These diagnostic variants are not production changes.
- Controls retain the exact production kernels. Coarse and sampled copies are regenerated from source with checked insertion anchors.
- Every implementation and diagnostic is verified against Eigen before timing at each thread count. Phase sums are checked against elapsed total; raw CSV samples are retained.
- Neither sampling nor equal numerical results repairs the pre-existing upstream numThreads data races. No hardware-counter claim is made from these clocks.
