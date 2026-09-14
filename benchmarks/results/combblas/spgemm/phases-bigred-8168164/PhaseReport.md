# SpGEMM phase measurements

Wall phases are measured outside the OpenMP loops. Tables show means so phase contributions add; total-call and diagnostic tables show medians. The sampled worker table is a separate experiment and shows shares of sampled active elapsed time across workers, not fractions of wall time. It excludes scheduling/wait time and may perturb very short columns.

## Whole-call controls and diagnostics

| Input | Threads | CSC control ms | DCSC control ms | Comb native ms | DCSC coarse/control | Comb coarse/control | Dense lookup ms | Static ms | Dense+static ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ER_V20000_D8 | 16 | 6.218 | 6.138 | 7.722 | 1.010 | 1.013 | 6.112 | 6.051 | 6.037 |
| ER_V20000_D8 | 128 | 4.112 | 3.920 | 7.433 | 1.076 | 0.974 | 3.362 | 3.737 | 3.717 |
| amazon0312.mtx | 16 | 86.360 | 88.659 | 255.985 | 0.988 | 1.008 | 87.383 | 227.313 | 225.199 |
| amazon0312.mtx | 128 | 54.620 | 54.725 | 99.113 | 1.003 | 1.011 | 49.971 | 83.979 | 81.982 |
| cant.mtx | 16 | 127.556 | 125.005 | 128.048 | 1.033 | 1.048 | 125.140 | 126.105 | 134.653 |
| cant.mtx | 128 | 56.099 | 57.802 | 54.187 | 0.974 | 1.014 | 52.277 | 42.673 | 42.834 |
| ecology1.mtx | 16 | 38.609 | 40.675 | 47.644 | 0.994 | 1.010 | 39.909 | 36.769 | 36.556 |
| ecology1.mtx | 128 | 51.056 | 55.208 | 47.771 | 1.005 | 0.994 | 51.270 | 42.242 | 41.656 |

## Wall-clock phases (mean milliseconds)

CombBLAS work and symbolic routines each build additional auxiliary lookup structures internally; their time stays in those routines. The outer lookup row is not the full cost of all CombBLAS lookup operations.

### ER_V20000_D8, 16 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.000 | 0.042 |
| Scratch/thread setup | 0.060 | 0.063 | 0.005 |
| Work bound/capacity | 0.180 | 0.190 | 0.285 |
| Symbolic | 1.125 | 1.078 | 1.316 |
| Prefix/numeric capacity | 0.040 | 0.024 | 0.013 |
| Output allocation/setup | 1.072 | 1.119 | 2.078 |
| Numeric incl. sort/write | 3.645 | 3.650 | 4.087 |
| Cleanup/free | 0.002 | 0.002 | 0.006 |
| Total | 6.125 | 6.127 | 7.831 |
### ER_V20000_D8, 128 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.000 | 0.044 |
| Scratch/thread setup | 0.070 | 0.081 | 0.046 |
| Work bound/capacity | 0.392 | 0.394 | 0.372 |
| Symbolic | 0.275 | 0.272 | 0.454 |
| Prefix/numeric capacity | 0.096 | 0.097 | 0.058 |
| Output allocation/setup | 1.930 | 2.033 | 4.971 |
| Numeric incl. sort/write | 1.092 | 1.047 | 1.277 |
| Cleanup/free | 0.002 | 0.002 | 0.026 |
| Total | 3.857 | 3.928 | 7.249 |
### amazon0312.mtx, 16 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.000 | 0.318 |
| Scratch/thread setup | 1.237 | 1.235 | 0.010 |
| Work bound/capacity | 4.367 | 4.580 | 10.268 |
| Symbolic | 16.094 | 16.080 | 47.971 |
| Prefix/numeric capacity | 0.543 | 0.543 | 0.242 |
| Output allocation/setup | 12.046 | 12.725 | 22.207 |
| Numeric incl. sort/write | 52.315 | 52.437 | 176.938 |
| Cleanup/free | 0.003 | 0.003 | 0.008 |
| Total | 86.606 | 87.604 | 257.961 |
### amazon0312.mtx, 128 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.000 | 0.846 |
| Scratch/thread setup | 1.555 | 1.613 | 0.056 |
| Work bound/capacity | 5.326 | 5.463 | 4.687 |
| Symbolic | 7.109 | 6.980 | 14.019 |
| Prefix/numeric capacity | 0.852 | 0.868 | 0.287 |
| Output allocation/setup | 15.562 | 16.455 | 29.269 |
| Numeric incl. sort/write | 22.948 | 22.577 | 51.046 |
| Cleanup/free | 0.004 | 0.004 | 0.029 |
| Total | 53.355 | 53.960 | 100.239 |
### cant.mtx, 16 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.000 | 0.054 |
| Scratch/thread setup | 0.137 | 0.136 | 0.008 |
| Work bound/capacity | 1.328 | 1.370 | 2.127 |
| Symbolic | 29.610 | 30.147 | 24.447 |
| Prefix/numeric capacity | 0.073 | 0.074 | 0.021 |
| Output allocation/setup | 14.181 | 14.233 | 25.715 |
| Numeric incl. sort/write | 84.418 | 83.180 | 81.831 |
| Cleanup/free | 0.003 | 0.002 | 0.006 |
| Total | 129.750 | 129.143 | 134.210 |
### cant.mtx, 128 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.000 | 0.084 |
| Scratch/thread setup | 0.335 | 0.314 | 0.050 |
| Work bound/capacity | 1.530 | 1.705 | 1.171 |
| Symbolic | 8.161 | 8.048 | 3.805 |
| Prefix/numeric capacity | 0.227 | 0.216 | 0.101 |
| Output allocation/setup | 20.998 | 20.961 | 34.579 |
| Numeric incl. sort/write | 26.429 | 26.187 | 15.298 |
| Cleanup/free | 0.003 | 0.003 | 0.027 |
| Total | 57.682 | 57.434 | 55.115 |
### ecology1.mtx, 16 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.000 | 0.851 |
| Scratch/thread setup | 1.764 | 1.780 | 0.009 |
| Work bound/capacity | 4.661 | 4.852 | 4.109 |
| Symbolic | 4.643 | 5.142 | 7.035 |
| Prefix/numeric capacity | 0.762 | 0.736 | 0.168 |
| Output allocation/setup | 11.025 | 12.504 | 19.755 |
| Numeric incl. sort/write | 15.785 | 15.433 | 16.191 |
| Cleanup/free | 0.005 | 0.005 | 0.007 |
| Total | 38.646 | 40.453 | 48.125 |
### ecology1.mtx, 128 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 0.001 | 2.002 |
| Scratch/thread setup | 4.517 | 4.773 | 0.058 |
| Work bound/capacity | 5.969 | 6.507 | 3.117 |
| Symbolic | 5.667 | 5.956 | 3.025 |
| Prefix/numeric capacity | 2.109 | 1.940 | 0.700 |
| Output allocation/setup | 16.837 | 21.102 | 29.547 |
| Numeric incl. sort/write | 15.719 | 15.299 | 9.168 |
| Cleanup/free | 0.008 | 0.009 | 0.032 |
| Total | 50.827 | 55.586 | 47.648 |

## Matched native tuple output

Both implementations use DCSC input and sorted std::tuple arrays. Allocation, initialization, symbolic/numeric work, sorting and destruction are included. Means reconcile by phase; controls quantify clock perturbation.

| Input | Threads | SpCraft control ms | Comb control ms | SpCraft coarse/control |
|---|---:|---:|---:|---:|
| ER_V20000_D8 | 16 | 6.660 | 7.722 | 1.026 |
| ER_V20000_D8 | 128 | 5.263 | 7.433 | 1.219 |
| amazon0312.mtx | 16 | 93.804 | 255.985 | 1.009 |
| amazon0312.mtx | 128 | 60.027 | 99.113 | 1.067 |
| cant.mtx | 16 | 129.374 | 128.048 | 1.077 |
| cant.mtx | 128 | 66.447 | 54.187 | 1.036 |
| ecology1.mtx | 16 | 47.852 | 47.644 | 1.001 |
| ecology1.mtx | 128 | 59.796 | 47.771 | 1.037 |
### Native tuples: ER_V20000_D8, 16 threads

| Step | SpCraft ms | CombBLAS ms |
|---|---:|---:|
| Lookup build | 0.000 | 0.042 |
| Scratch/thread setup | 0.070 | 0.005 |
| Work bound/capacity | 0.180 | 0.285 |
| Symbolic | 1.072 | 1.316 |
| Prefix/numeric capacity | 0.024 | 0.013 |
| Output allocation/setup | 1.805 | 2.078 |
| Numeric incl. sort/write | 3.684 | 4.087 |
| Cleanup/free | 0.002 | 0.006 |
| Total | 6.835 | 7.831 |
### Native tuples: ER_V20000_D8, 128 threads

| Step | SpCraft ms | CombBLAS ms |
|---|---:|---:|
| Lookup build | 0.000 | 0.044 |
| Scratch/thread setup | 0.094 | 0.046 |
| Work bound/capacity | 0.391 | 0.372 |
| Symbolic | 0.272 | 0.454 |
| Prefix/numeric capacity | 0.099 | 0.058 |
| Output allocation/setup | 4.532 | 4.971 |
| Numeric incl. sort/write | 1.140 | 1.277 |
| Cleanup/free | 0.001 | 0.026 |
| Total | 6.531 | 7.249 |
### Native tuples: amazon0312.mtx, 16 threads

| Step | SpCraft ms | CombBLAS ms |
|---|---:|---:|
| Lookup build | 0.000 | 0.318 |
| Scratch/thread setup | 1.254 | 0.010 |
| Work bound/capacity | 4.476 | 10.268 |
| Symbolic | 16.350 | 47.971 |
| Prefix/numeric capacity | 0.536 | 0.242 |
| Output allocation/setup | 18.386 | 22.207 |
| Numeric incl. sort/write | 53.711 | 176.938 |
| Cleanup/free | 0.002 | 0.008 |
| Total | 94.716 | 257.961 |
### Native tuples: amazon0312.mtx, 128 threads

| Step | SpCraft ms | CombBLAS ms |
|---|---:|---:|
| Lookup build | 0.000 | 0.846 |
| Scratch/thread setup | 1.813 | 0.056 |
| Work bound/capacity | 5.125 | 4.687 |
| Symbolic | 6.197 | 14.019 |
| Prefix/numeric capacity | 0.862 | 0.287 |
| Output allocation/setup | 24.794 | 29.269 |
| Numeric incl. sort/write | 24.546 | 51.046 |
| Cleanup/free | 0.003 | 0.029 |
| Total | 63.341 | 100.239 |
### Native tuples: cant.mtx, 16 threads

| Step | SpCraft ms | CombBLAS ms |
|---|---:|---:|
| Lookup build | 0.000 | 0.054 |
| Scratch/thread setup | 0.140 | 0.008 |
| Work bound/capacity | 1.361 | 2.127 |
| Symbolic | 33.511 | 24.447 |
| Prefix/numeric capacity | 0.072 | 0.021 |
| Output allocation/setup | 23.132 | 25.715 |
| Numeric incl. sort/write | 81.151 | 81.831 |
| Cleanup/free | 0.002 | 0.006 |
| Total | 139.369 | 134.210 |
### Native tuples: cant.mtx, 128 threads

| Step | SpCraft ms | CombBLAS ms |
|---|---:|---:|
| Lookup build | 0.000 | 0.084 |
| Scratch/thread setup | 0.444 | 0.050 |
| Work bound/capacity | 1.777 | 1.171 |
| Symbolic | 7.338 | 3.805 |
| Prefix/numeric capacity | 0.207 | 0.101 |
| Output allocation/setup | 31.028 | 34.579 |
| Numeric incl. sort/write | 28.175 | 15.298 |
| Cleanup/free | 0.002 | 0.027 |
| Total | 68.972 | 55.115 |
### Native tuples: ecology1.mtx, 16 threads

| Step | SpCraft ms | CombBLAS ms |
|---|---:|---:|
| Lookup build | 0.000 | 0.851 |
| Scratch/thread setup | 1.840 | 0.009 |
| Work bound/capacity | 4.843 | 4.109 |
| Symbolic | 5.449 | 7.035 |
| Prefix/numeric capacity | 0.729 | 0.168 |
| Output allocation/setup | 16.648 | 19.755 |
| Numeric incl. sort/write | 18.356 | 16.191 |
| Cleanup/free | 0.004 | 0.007 |
| Total | 47.870 | 48.125 |
### Native tuples: ecology1.mtx, 128 threads

| Step | SpCraft ms | CombBLAS ms |
|---|---:|---:|
| Lookup build | 0.001 | 2.002 |
| Scratch/thread setup | 5.911 | 0.058 |
| Work bound/capacity | 7.027 | 3.117 |
| Symbolic | 6.019 | 3.025 |
| Prefix/numeric capacity | 2.323 | 0.700 |
| Output allocation/setup | 24.288 | 29.547 |
| Numeric incl. sort/write | 17.199 | 9.168 |
| Cleanup/free | 0.008 | 0.032 |
| Total | 62.778 | 47.648 |

## Sampled numeric worker time

DCSC hashing includes repeated left-column lookup; CombBLAS lookup is timed separately before hashing. Percentages normalize the summed sampled durations. Sampling selects approximately 8,192 evenly spaced candidate columns per call (all columns for smaller inputs); these shares are descriptive, not estimates of a critical thread or an unbiased distribution of irregular work.

| Input | Threads | Backend | Lookup % | Alloc/init % | Hash/expand % | Compact % | Sort % | Write % | Free % | Sampled/coarse total |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ER_V20000_D8 | 16 | DCSC | 0.0 | 3.6 | 32.1 | 10.1 | 48.6 | 3.6 | 2.0 | 0.987 |
| ER_V20000_D8 | 16 | CombBLAS | 6.0 | 4.6 | 27.8 | 9.8 | 45.1 | 4.9 | 1.8 | 1.011 |
| ER_V20000_D8 | 128 | DCSC | 0.0 | 4.4 | 19.2 | 4.8 | 23.0 | 32.0 | 16.6 | 0.811 |
| ER_V20000_D8 | 128 | CombBLAS | 23.0 | 3.4 | 16.6 | 5.9 | 26.4 | 18.5 | 6.2 | 0.811 |
| amazon0312.mtx | 16 | DCSC | 0.0 | 3.4 | 46.2 | 8.2 | 36.0 | 4.0 | 2.1 | 1.002 |
| amazon0312.mtx | 16 | CombBLAS | 10.0 | 4.1 | 40.7 | 7.5 | 32.3 | 3.5 | 1.9 | 1.001 |
| amazon0312.mtx | 128 | DCSC | 0.0 | 2.6 | 70.6 | 2.8 | 12.1 | 8.1 | 3.8 | 0.959 |
| amazon0312.mtx | 128 | CombBLAS | 14.8 | 2.8 | 43.7 | 4.8 | 20.8 | 9.0 | 4.2 | 0.960 |
| cant.mtx | 16 | DCSC | 0.0 | 1.4 | 69.9 | 2.4 | 24.7 | 1.2 | 0.3 | 0.999 |
| cant.mtx | 16 | CombBLAS | 2.7 | 2.1 | 64.9 | 3.0 | 25.1 | 1.9 | 0.3 | 1.000 |
| cant.mtx | 128 | DCSC | 0.0 | 1.2 | 73.6 | 1.5 | 10.3 | 10.9 | 2.5 | 1.057 |
| cant.mtx | 128 | CombBLAS | 4.2 | 1.9 | 56.4 | 2.6 | 21.6 | 12.0 | 1.3 | 0.918 |
| ecology1.mtx | 16 | DCSC | 0.0 | 10.0 | 33.2 | 11.5 | 20.5 | 13.2 | 11.6 | 0.998 |
| ecology1.mtx | 16 | CombBLAS | 18.6 | 11.8 | 23.2 | 11.0 | 16.9 | 9.2 | 9.3 | 1.000 |
| ecology1.mtx | 128 | DCSC | 0.0 | 0.7 | 76.8 | 0.4 | 1.7 | 14.1 | 6.2 | 0.960 |
| ecology1.mtx | 128 | CombBLAS | 27.6 | 17.9 | 20.2 | 5.0 | 13.2 | 6.4 | 9.7 | 0.900 |

## Tuple-to-DCSC conversion (mean milliseconds)

| Input | Threads | Native call | Compression | DCSC free | Tuple free |
|---|---:|---:|---:|---:|---:|
| ER_V20000_D8 | 16 | 7.323 | 6.045 | 0.003 | 0.001 |
| ER_V20000_D8 | 128 | 5.702 | 6.580 | 0.002 | 0.001 |
| amazon0312.mtx | 16 | 256.129 | 54.842 | 0.003 | 0.001 |
| amazon0312.mtx | 128 | 96.736 | 59.800 | 0.003 | 0.001 |
| cant.mtx | 16 | 127.980 | 65.325 | 0.003 | 0.001 |
| cant.mtx | 128 | 52.253 | 73.748 | 0.002 | 0.001 |
| ecology1.mtx | 16 | 47.921 | 48.799 | 0.003 | 0.001 |
| ecology1.mtx | 128 | 43.699 | 78.711 | 0.003 | 0.001 |

## Interpretation limits

- Dense lookup is valid only when sorted unique DCSC column IDs cover every logical column (nzc==n). Otherwise it falls back to the original lookup.
- Static and dense+static change scheduling only, or scheduling plus that lookup shortcut. These diagnostic variants are not production changes.
- Controls retain the exact production kernels. Coarse and sampled copies are regenerated from source with checked insertion anchors.
- Every implementation and diagnostic is verified against Eigen before timing at each thread count. Phase sums are checked against elapsed total; raw CSV samples are retained.
- Neither sampling nor equal numerical results repairs the pre-existing upstream numThreads data races. No hardware-counter claim is made from these clocks.
