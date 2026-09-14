# SpCraft / CombBLAS hash SpGEMM comparison

Each result is A². SpCraft computes CSR A×A; CombBLAS computes DCSC Aᵀ×Aᵀ with `LocalSpGEMMHash(..., sort=true)`. Input conversion and Eigen verification are outside timing. Each native call includes symbolic work, numeric work, sorting, allocation, and destruction. Native output formats differ: CSR versus tuples.

Ratio = SpCraft median / CombBLAS median; below 1 means SpCraft is faster. Intervals are 95% paired bootstrap intervals over alternating AB/BA samples. CV measures sample variability, not the performance gap. Few samples or a noisy shared machine limit the conclusions.

| Matrix | Types | Threads | Pairs | SpCraft ms | CombBLAS ms | Ratio [95% CI] | CV %, SpCraft / CombBLAS |
|---|---|---:|---:|---:|---:|---|---|
| amazon0312.mtx squared | FP64/int32 | 1 | 20 | 1073.416 | 1287.811 | 0.834 [0.833, 0.834] | 0.1 / 0.1 |
| amazon0312.mtx squared | FP64/int32 | 4 | 21 | 284.397 | 394.733 | 0.720 [0.720, 0.722] | 0.2 / 0.1 |
| amazon0312.mtx squared | FP64/int32 | 16 | 21 | 85.348 | 111.897 | 0.763 [0.761, 0.764] | 0.3 / 0.1 |
| amazon0312.mtx squared | FP64/int32 | 32 | 21 | 55.022 | 69.683 | 0.790 [0.787, 0.795] | 0.3 / 0.6 |
| amazon0312.mtx squared | FP64/int32 | 64 | 21 | 46.731 | 55.931 | 0.836 [0.822, 0.852] | 1.1 / 1.3 |
| amazon0312.mtx squared | FP64/int32 | 128 | 21 | 54.694 | 61.007 | 0.897 [0.819, 0.995] | 4.5 / 6.0 |
| cant.mtx squared | FP64/int32 | 1 | 14 | 1743.163 | 1617.832 | 1.077 [1.073, 1.083] | 0.7 / 0.2 |
| cant.mtx squared | FP64/int32 | 4 | 21 | 449.545 | 435.469 | 1.032 [1.031, 1.034] | 0.3 / 0.2 |
| cant.mtx squared | FP64/int32 | 16 | 21 | 128.212 | 133.481 | 0.961 [0.957, 0.962] | 0.3 / 0.2 |
| cant.mtx squared | FP64/int32 | 32 | 21 | 75.356 | 83.283 | 0.905 [0.895, 0.907] | 0.8 / 0.4 |
| cant.mtx squared | FP64/int32 | 64 | 21 | 54.069 | 64.364 | 0.840 [0.828, 0.850] | 1.2 / 1.4 |
| cant.mtx squared | FP64/int32 | 128 | 21 | 60.358 | 54.696 | 1.104 [0.996, 1.180] | 4.1 / 5.4 |
| ecology1.mtx squared | FP64/int32 | 1 | 21 | 252.053 | 377.759 | 0.667 [0.665, 0.668] | 0.2 / 0.3 |
| ecology1.mtx squared | FP64/int32 | 4 | 21 | 84.870 | 137.086 | 0.619 [0.615, 0.621] | 1.9 / 1.2 |
| ecology1.mtx squared | FP64/int32 | 16 | 21 | 42.753 | 68.194 | 0.627 [0.623, 0.629] | 0.6 / 0.6 |
| ecology1.mtx squared | FP64/int32 | 32 | 21 | 45.832 | 56.700 | 0.808 [0.798, 0.814] | 1.4 / 0.6 |
| ecology1.mtx squared | FP64/int32 | 64 | 21 | 52.666 | 49.804 | 1.057 [1.025, 1.110] | 3.0 / 2.5 |
| ecology1.mtx squared | FP64/int32 | 128 | 21 | 62.142 | 50.651 | 1.227 [1.216, 1.363] | 4.1 / 6.5 |
| ER_V20000_D32 squared | FP64/int32 | 1 | 17 | 1295.688 | 1364.668 | 0.949 [0.949, 0.951] | 0.1 / 0.1 |
| ER_V20000_D32 squared | FP64/int32 | 4 | 21 | 339.443 | 365.704 | 0.928 [0.927, 0.929] | 0.2 / 0.2 |
| ER_V20000_D32 squared | FP64/int32 | 16 | 21 | 100.997 | 114.174 | 0.885 [0.883, 0.889] | 0.2 / 0.4 |
| ER_V20000_D32 squared | FP64/int32 | 32 | 21 | 61.892 | 74.011 | 0.836 [0.831, 0.844] | 0.4 / 1.1 |
| ER_V20000_D32 squared | FP64/int32 | 64 | 21 | 45.442 | 58.778 | 0.773 [0.764, 0.804] | 0.4 / 2.7 |
| ER_V20000_D32 squared | FP64/int32 | 128 | 21 | 45.606 | 63.393 | 0.719 [0.632, 0.875] | 6.6 / 10.3 |
| ER_V20000_D8 squared | FP64/int32 | 1 | 21 | 75.331 | 84.147 | 0.895 [0.881, 0.902] | 0.9 / 0.8 |
| ER_V20000_D8 squared | FP64/int32 | 4 | 21 | 20.231 | 22.877 | 0.884 [0.864, 0.892] | 1.5 / 0.5 |
| ER_V20000_D8 squared | FP64/int32 | 16 | 21 | 6.095 | 7.518 | 0.811 [0.795, 0.875] | 1.0 / 5.0 |
| ER_V20000_D8 squared | FP64/int32 | 32 | 21 | 3.941 | 5.251 | 0.751 [0.738, 0.847] | 3.1 / 8.9 |
| ER_V20000_D8 squared | FP64/int32 | 64 | 21 | 3.027 | 5.040 | 0.601 [0.586, 0.812] | 8.2 / 17.5 |
| ER_V20000_D8 squared | FP64/int32 | 128 | 21 | 3.564 | 6.926 | 0.515 [0.512, 0.777] | 10.5 / 22.5 |
| RMAT_S12_E8 squared | FP64/int32 | 1 | 21 | 178.088 | 183.648 | 0.970 [0.969, 0.970] | 0.1 / 0.2 |
| RMAT_S12_E8 squared | FP64/int32 | 4 | 21 | 46.341 | 73.558 | 0.630 [0.629, 0.630] | 0.1 / 0.2 |
| RMAT_S12_E8 squared | FP64/int32 | 16 | 21 | 13.098 | 29.340 | 0.446 [0.444, 0.461] | 2.0 / 1.5 |
| RMAT_S12_E8 squared | FP64/int32 | 32 | 21 | 7.757 | 19.254 | 0.403 [0.400, 0.432] | 3.9 / 3.1 |
| RMAT_S12_E8 squared | FP64/int32 | 64 | 21 | 5.525 | 14.300 | 0.386 [0.375, 0.434] | 7.1 / 6.9 |
| RMAT_S12_E8 squared | FP64/int32 | 128 | 21 | 5.239 | 12.704 | 0.412 [0.408, 0.533] | 10.9 / 15.7 |

Maximum verification error: 3.72529e-09.

Raw per-iteration samples and build/runtime metadata accompany this report.
