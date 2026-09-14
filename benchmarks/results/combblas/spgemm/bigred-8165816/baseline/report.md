# SpCraft / CombBLAS hash SpGEMM comparison

Each result is A². SpCraft computes CSR A×A; CombBLAS computes DCSC Aᵀ×Aᵀ with `LocalSpGEMMHash(..., sort=true)`. Input conversion and Eigen verification are outside timing. Each native call includes symbolic work, numeric work, sorting, allocation, and destruction. Native output formats differ: CSR versus tuples.

Ratio = SpCraft median / CombBLAS median; below 1 means SpCraft is faster. Intervals are 95% paired bootstrap intervals over alternating AB/BA samples. CV measures sample variability, not the performance gap. Few samples or a noisy shared machine limit the conclusions.

| Matrix | Types | Threads | Pairs | SpCraft ms | CombBLAS ms | Ratio [95% CI] | CV %, SpCraft / CombBLAS |
|---|---|---:|---:|---:|---:|---|---|
| amazon0312.mtx squared | FP64/int32 | 1 | 21 | 1029.566 | 1202.584 | 0.856 [0.856, 0.856] | 0.1 / 0.0 |
| amazon0312.mtx squared | FP64/int32 | 4 | 21 | 296.080 | 363.565 | 0.814 [0.814, 0.815] | 0.1 / 0.1 |
| amazon0312.mtx squared | FP64/int32 | 16 | 21 | 116.019 | 107.515 | 1.079 [1.078, 1.080] | 0.1 / 0.2 |
| amazon0312.mtx squared | FP64/int32 | 32 | 21 | 84.350 | 68.912 | 1.224 [1.222, 1.226] | 0.2 / 0.4 |
| amazon0312.mtx squared | FP64/int32 | 64 | 21 | 76.063 | 57.895 | 1.314 [1.308, 1.412] | 0.7 / 4.1 |
| amazon0312.mtx squared | FP64/int32 | 128 | 21 | 93.419 | 80.477 | 1.161 [1.160, 1.711] | 5.2 / 19.8 |
| cant.mtx squared | FP64/int32 | 1 | 14 | 1677.458 | 1637.002 | 1.025 [1.024, 1.026] | 0.0 / 0.1 |
| cant.mtx squared | FP64/int32 | 4 | 21 | 515.811 | 437.009 | 1.180 [1.179, 1.182] | 0.1 / 0.2 |
| cant.mtx squared | FP64/int32 | 16 | 21 | 149.007 | 132.625 | 1.124 [1.121, 1.126] | 0.1 / 0.3 |
| cant.mtx squared | FP64/int32 | 32 | 21 | 84.829 | 82.714 | 1.026 [1.019, 1.042] | 0.2 / 1.1 |
| cant.mtx squared | FP64/int32 | 64 | 21 | 67.843 | 64.973 | 1.044 [1.003, 1.084] | 1.7 / 3.1 |
| cant.mtx squared | FP64/int32 | 128 | 21 | 87.615 | 59.639 | 1.469 [1.358, 1.710] | 4.5 / 8.8 |
| ecology1.mtx squared | FP64/int32 | 1 | 21 | 266.894 | 382.111 | 0.698 [0.696, 0.700] | 0.3 / 0.5 |
| ecology1.mtx squared | FP64/int32 | 4 | 21 | 130.349 | 120.688 | 1.080 [1.078, 1.086] | 1.2 / 1.2 |
| ecology1.mtx squared | FP64/int32 | 16 | 21 | 145.406 | 49.899 | 2.914 [2.886, 2.928] | 0.2 / 1.4 |
| ecology1.mtx squared | FP64/int32 | 32 | 21 | 132.336 | 44.945 | 2.944 [2.930, 2.954] | 0.1 / 0.7 |
| ecology1.mtx squared | FP64/int32 | 64 | 21 | 131.158 | 43.712 | 3.001 [2.980, 3.153] | 0.6 / 3.4 |
| ecology1.mtx squared | FP64/int32 | 128 | 21 | 166.221 | 51.559 | 3.224 [3.224, 4.012] | 2.3 / 12.7 |
| ER_V20000_D32 squared | FP64/int32 | 1 | 18 | 1279.254 | 1350.959 | 0.947 [0.946, 0.947] | 0.1 / 0.1 |
| ER_V20000_D32 squared | FP64/int32 | 4 | 21 | 331.116 | 359.486 | 0.921 [0.920, 0.923] | 0.2 / 0.2 |
| ER_V20000_D32 squared | FP64/int32 | 16 | 21 | 96.848 | 111.351 | 0.870 [0.868, 0.877] | 0.1 / 0.7 |
| ER_V20000_D32 squared | FP64/int32 | 32 | 21 | 58.733 | 71.632 | 0.820 [0.812, 0.832] | 0.8 / 1.6 |
| ER_V20000_D32 squared | FP64/int32 | 64 | 21 | 42.837 | 58.725 | 0.729 [0.727, 0.792] | 2.4 / 4.8 |
| ER_V20000_D32 squared | FP64/int32 | 128 | 21 | 45.859 | 68.449 | 0.670 [0.624, 0.925] | 4.1 / 16.1 |
| ER_V20000_D8 squared | FP64/int32 | 1 | 21 | 74.285 | 82.718 | 0.898 [0.891, 0.900] | 0.4 / 0.2 |
| ER_V20000_D8 squared | FP64/int32 | 4 | 21 | 20.219 | 22.146 | 0.913 [0.903, 0.915] | 0.6 / 0.2 |
| ER_V20000_D8 squared | FP64/int32 | 16 | 21 | 7.370 | 8.510 | 0.866 [0.854, 1.059] | 3.6 / 11.3 |
| ER_V20000_D8 squared | FP64/int32 | 32 | 21 | 5.305 | 6.619 | 0.802 [0.793, 1.147] | 5.3 / 19.0 |
| ER_V20000_D8 squared | FP64/int32 | 64 | 21 | 4.465 | 6.292 | 0.710 [0.699, 1.201] | 7.2 / 26.9 |
| ER_V20000_D8 squared | FP64/int32 | 128 | 21 | 5.225 | 7.884 | 0.663 [0.652, 1.222] | 9.0 / 30.9 |
| RMAT_S12_E8 squared | FP64/int32 | 1 | 21 | 177.135 | 183.001 | 0.968 [0.967, 0.969] | 0.1 / 0.2 |
| RMAT_S12_E8 squared | FP64/int32 | 4 | 21 | 46.009 | 72.979 | 0.630 [0.630, 0.631] | 0.1 / 0.1 |
| RMAT_S12_E8 squared | FP64/int32 | 16 | 21 | 13.158 | 29.964 | 0.439 [0.438, 0.462] | 2.8 / 2.5 |
| RMAT_S12_E8 squared | FP64/int32 | 32 | 21 | 7.813 | 19.837 | 0.394 [0.393, 0.432] | 5.1 / 4.4 |
| RMAT_S12_E8 squared | FP64/int32 | 64 | 21 | 5.443 | 14.661 | 0.371 [0.370, 0.428] | 7.4 / 7.9 |
| RMAT_S12_E8 squared | FP64/int32 | 128 | 21 | 5.644 | 12.587 | 0.448 [0.403, 0.559] | 10.7 / 12.9 |

Maximum verification error: 3.72529e-09.

Raw per-iteration samples and build/runtime metadata accompany this report.
