# Paired native tuple comparison

Both kernels read identical DCSC inputs and return column/row-sorted std::tuple<IT,IT,NT> arrays. Timings include initialization/allocation, symbolic, numeric, sorting and destruction. No output conversion occurs. Full independent Eigen checks precede timing at each thread count. Only these two kernels run during warmup and timing. Three warmup rounds precede measured pairs, whose order alternates AB/BA.

Ratio = SpCraft / CombBLAS median. Below one favors SpCraft. Intervals are paired bootstrap 95% intervals over rounds, not uncertainty across node allocations. CV describes sample variability separately from the gap.

| Matrix | Threads | Pairs | SpCraft ms | CombBLAS ms | Ratio [95% CI] | CV % (Sp / Comb) |
|---|---:|---:|---:|---:|---|---|
| amazon0312.mtx squared | 1 | 19 | 1235.860 | 1245.516 | 0.992 [0.992, 0.993] | 0.0 / 0.0 |
| amazon0312.mtx squared | 4 | 20 | 672.692 | 679.037 | 0.991 [0.990, 0.991] | 0.1 / 0.1 |
| amazon0312.mtx squared | 16 | 20 | 244.248 | 246.173 | 0.992 [0.992, 0.993] | 0.1 / 0.1 |
| amazon0312.mtx squared | 32 | 20 | 157.842 | 158.187 | 0.998 [0.996, 0.999] | 0.1 / 0.1 |
| amazon0312.mtx squared | 64 | 20 | 113.106 | 112.735 | 1.003 [0.996, 1.010] | 0.4 / 0.5 |
| amazon0312.mtx squared | 128 | 20 | 94.459 | 93.578 | 1.009 [0.995, 1.017] | 0.9 / 0.7 |
| cant.mtx squared | 1 | 14 | 1607.227 | 1623.237 | 0.990 [0.989, 0.991] | 0.1 / 0.2 |
| cant.mtx squared | 4 | 20 | 426.534 | 431.541 | 0.988 [0.987, 0.989] | 2.6 / 2.3 |
| cant.mtx squared | 16 | 20 | 127.298 | 127.690 | 0.997 [0.996, 0.998] | 0.2 / 0.2 |
| cant.mtx squared | 32 | 20 | 78.011 | 78.345 | 0.996 [0.994, 0.998] | 0.2 / 0.2 |
| cant.mtx squared | 64 | 20 | 58.825 | 58.721 | 1.002 [0.986, 1.015] | 1.6 / 1.5 |
| cant.mtx squared | 128 | 20 | 47.876 | 47.519 | 1.008 [0.998, 1.018] | 1.1 / 1.2 |
| ecology1.mtx squared | 1 | 20 | 380.049 | 373.518 | 1.017 [1.017, 1.019] | 0.2 / 0.2 |
| ecology1.mtx squared | 4 | 20 | 119.951 | 116.029 | 1.034 [1.031, 1.046] | 1.0 / 1.3 |
| ecology1.mtx squared | 16 | 20 | 51.784 | 47.868 | 1.082 [1.072, 1.089] | 0.9 / 1.0 |
| ecology1.mtx squared | 32 | 20 | 46.998 | 43.112 | 1.090 [1.085, 1.097] | 1.0 / 0.6 |
| ecology1.mtx squared | 64 | 20 | 45.310 | 40.599 | 1.116 [1.029, 1.175] | 3.8 / 3.4 |
| ecology1.mtx squared | 128 | 20 | 43.713 | 39.066 | 1.119 [1.103, 1.128] | 1.5 / 0.8 |
| ER_V20000_D32 squared | 1 | 17 | 1329.458 | 1346.550 | 0.987 [0.987, 0.988] | 0.1 / 0.1 |
| ER_V20000_D32 squared | 4 | 20 | 353.974 | 358.597 | 0.987 [0.987, 0.988] | 0.1 / 0.1 |
| ER_V20000_D32 squared | 16 | 20 | 109.341 | 110.003 | 0.994 [0.992, 0.996] | 0.4 / 0.2 |
| ER_V20000_D32 squared | 32 | 20 | 69.779 | 69.952 | 0.998 [0.995, 1.002] | 0.8 / 0.4 |
| ER_V20000_D32 squared | 64 | 20 | 54.175 | 53.918 | 1.005 [0.991, 1.021] | 0.8 / 2.0 |
| ER_V20000_D32 squared | 128 | 20 | 58.598 | 55.243 | 1.061 [0.837, 1.154] | 8.1 / 8.7 |
| ER_V20000_D8 squared | 1 | 20 | 82.648 | 82.515 | 1.002 [0.997, 1.005] | 0.4 / 0.3 |
| ER_V20000_D8 squared | 4 | 20 | 22.234 | 22.090 | 1.007 [1.000, 1.011] | 0.7 / 0.3 |
| ER_V20000_D8 squared | 16 | 20 | 7.045 | 6.972 | 1.010 [0.992, 1.032] | 1.3 / 1.1 |
| ER_V20000_D8 squared | 32 | 20 | 4.692 | 4.580 | 1.024 [0.997, 1.043] | 2.0 / 1.4 |
| ER_V20000_D8 squared | 64 | 20 | 3.867 | 3.737 | 1.035 [0.983, 1.058] | 2.9 / 1.8 |
| ER_V20000_D8 squared | 128 | 20 | 4.300 | 4.201 | 1.024 [0.993, 1.034] | 2.6 / 1.9 |
| ER_V20000_D8 squared stride64 | 1 | 20 | 113.105 | 115.824 | 0.977 [0.973, 0.980] | 0.3 / 0.2 |
| ER_V20000_D8 squared stride64 | 4 | 20 | 29.981 | 30.629 | 0.979 [0.975, 0.982] | 0.4 / 0.4 |
| ER_V20000_D8 squared stride64 | 16 | 20 | 9.078 | 9.208 | 0.986 [0.960, 1.008] | 1.4 / 1.4 |
| ER_V20000_D8 squared stride64 | 32 | 20 | 5.776 | 5.802 | 0.996 [0.963, 1.019] | 1.8 / 1.8 |
| ER_V20000_D8 squared stride64 | 64 | 20 | 4.357 | 4.322 | 1.008 [0.959, 1.031] | 2.6 / 3.0 |
| ER_V20000_D8 squared stride64 | 128 | 20 | 4.626 | 4.661 | 0.993 [0.963, 1.009] | 2.1 / 1.0 |
| RMAT_S12_E8 squared | 1 | 20 | 179.968 | 181.365 | 0.992 [0.992, 0.993] | 0.2 / 0.1 |
| RMAT_S12_E8 squared | 4 | 20 | 71.651 | 72.121 | 0.993 [0.992, 0.995] | 0.2 / 0.2 |
| RMAT_S12_E8 squared | 16 | 20 | 28.152 | 28.208 | 0.998 [0.997, 1.000] | 0.3 / 0.2 |
| RMAT_S12_E8 squared | 32 | 20 | 18.064 | 18.095 | 0.998 [0.994, 1.003] | 0.5 / 0.5 |
| RMAT_S12_E8 squared | 64 | 20 | 12.581 | 12.511 | 1.006 [0.999, 1.009] | 0.9 / 0.9 |
| RMAT_S12_E8 squared | 128 | 20 | 9.793 | 9.815 | 0.998 [0.989, 1.015] | 1.5 / 2.7 |

Maximum absolute verification error: 0.
