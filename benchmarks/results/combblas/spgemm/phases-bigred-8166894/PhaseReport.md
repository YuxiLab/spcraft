# SpGEMM phase measurements

Wall phases are measured outside the OpenMP loops. Tables show means so phase contributions add; total-call and diagnostic tables show medians. The sampled worker table is a separate experiment and shows shares of sampled active elapsed time across workers, not fractions of wall time. It excludes scheduling/wait time and may perturb very short columns.

## Whole-call controls and diagnostics

| Input | Threads | CSC control ms | DCSC control ms | Comb native ms | DCSC coarse/control | Comb coarse/control | Dense lookup ms | Static ms | Dense+static ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ecology1.mtx | 1 | 246.913 | 367.461 | 373.912 | 1.007 | 1.036 | 259.879 | 377.836 | 251.392 |
| ecology1.mtx | 16 | 42.551 | 62.047 | 47.820 | 0.996 | 0.999 | 44.048 | 59.878 | 42.052 |
| ecology1.mtx | 128 | 60.046 | 79.315 | 47.350 | 0.998 | 1.029 | 58.649 | 63.845 | 52.280 |

## Wall-clock phases (mean milliseconds)

CombBLAS work and symbolic routines each build additional auxiliary lookup structures internally; their time stays in those routines. The outer lookup row is not the full cost of all CombBLAS lookup operations.

### ecology1.mtx, 1 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 13.916 | 0.798 |
| Scratch/thread setup | 1.257 | 1.332 | 0.003 |
| Work bound/capacity | 10.502 | 39.470 | 46.312 |
| Symbolic | 62.267 | 93.783 | 91.924 |
| Prefix/numeric capacity | 2.400 | 2.382 | 0.544 |
| Output allocation/setup | 11.063 | 12.214 | 18.631 |
| Numeric incl. sort/write | 172.535 | 206.649 | 228.655 |
| Cleanup/free | 0.002 | 0.003 | 0.005 |
| Total | 260.027 | 369.749 | 386.872 |
### ecology1.mtx, 16 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 13.952 | 0.800 |
| Scratch/thread setup | 1.447 | 1.461 | 0.008 |
| Work bound/capacity | 5.187 | 5.926 | 4.013 |
| Symbolic | 4.650 | 6.932 | 7.068 |
| Prefix/numeric capacity | 5.270 | 4.752 | 0.151 |
| Output allocation/setup | 11.248 | 12.485 | 19.822 |
| Numeric incl. sort/write | 15.155 | 16.339 | 15.895 |
| Cleanup/free | 0.004 | 0.004 | 0.008 |
| Total | 42.962 | 61.851 | 47.766 |
### ecology1.mtx, 128 threads

| Step | CSC ms | DCSC ms | CombBLAS native ms |
|---|---:|---:|---:|
| Lookup build | 0.000 | 15.252 | 1.684 |
| Scratch/thread setup | 4.530 | 2.600 | 0.049 |
| Work bound/capacity | 5.783 | 7.352 | 3.050 |
| Symbolic | 5.558 | 7.158 | 2.950 |
| Prefix/numeric capacity | 12.528 | 11.890 | 0.328 |
| Output allocation/setup | 17.119 | 18.511 | 32.013 |
| Numeric incl. sort/write | 14.962 | 16.590 | 8.834 |
| Cleanup/free | 0.008 | 0.008 | 0.029 |
| Total | 60.489 | 79.362 | 48.938 |

## Sampled numeric worker time

DCSC hashing includes repeated left-column lookup; CombBLAS lookup is timed separately before hashing. Percentages normalize the summed sampled durations. Sampling selects approximately 8,192 evenly spaced candidate columns per call (all columns for smaller inputs); these shares are descriptive, not estimates of a critical thread or an unbiased distribution of irregular work.

| Input | Threads | Backend | Lookup % | Alloc/init % | Hash/expand % | Compact % | Sort % | Write % | Free % | Sampled/coarse total |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ecology1.mtx | 1 | DCSC | 0.0 | 10.3 | 34.5 | 13.9 | 19.4 | 11.5 | 10.4 | 1.008 |
| ecology1.mtx | 1 | CombBLAS | 17.4 | 11.8 | 22.3 | 11.5 | 17.7 | 9.5 | 9.7 | 1.005 |
| ecology1.mtx | 16 | DCSC | 0.0 | 10.6 | 39.2 | 12.2 | 17.4 | 10.4 | 10.2 | 1.004 |
| ecology1.mtx | 16 | CombBLAS | 18.7 | 12.0 | 22.1 | 11.2 | 17.5 | 9.3 | 9.3 | 1.000 |
| ecology1.mtx | 128 | DCSC | 0.0 | 0.4 | 81.3 | 0.4 | 1.2 | 10.5 | 6.2 | 0.990 |
| ecology1.mtx | 128 | CombBLAS | 28.4 | 17.8 | 17.8 | 5.4 | 12.8 | 6.8 | 11.2 | 0.868 |

## Tuple-to-DCSC conversion (mean milliseconds)

| Input | Threads | Native call | Compression | DCSC free | Tuple free |
|---|---:|---:|---:|---:|---:|
| ecology1.mtx | 1 | 372.547 | 46.478 | 0.004 | 0.001 |
| ecology1.mtx | 16 | 48.059 | 48.770 | 0.003 | 0.001 |
| ecology1.mtx | 128 | 42.402 | 83.097 | 0.003 | 0.001 |

## Interpretation limits

- Dense lookup is valid only when sorted unique DCSC column IDs cover every logical column (nzc==n). Otherwise it falls back to the original lookup.
- Static and dense+static change scheduling only, or scheduling plus that lookup shortcut. These diagnostic variants are not production changes.
- Controls retain the exact production kernels. Coarse and sampled copies are regenerated from source with checked insertion anchors.
- Every implementation and diagnostic is verified against Eigen before timing at each thread count. Phase sums are checked against elapsed total; raw CSV samples are retained.
- Neither sampling nor equal numerical results repairs the pre-existing upstream numThreads data races. No hardware-counter claim is made from these clocks.
