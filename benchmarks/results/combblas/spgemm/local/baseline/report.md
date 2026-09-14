# SpCraft / CombBLAS hash SpGEMM comparison

Each result is A². SpCraft computes CSR A×A; CombBLAS computes DCSC Aᵀ×Aᵀ with `LocalSpGEMMHash(..., sort=true)`. Input conversion and Eigen verification are outside timing. Each native call includes symbolic work, numeric work, sorting, allocation, and destruction. Native output formats differ: CSR versus tuples.

Ratio = SpCraft median / CombBLAS median; below 1 means SpCraft is faster. Intervals are 95% paired bootstrap intervals over alternating AB/BA samples. CV measures sample variability, not the performance gap. Few samples or a noisy shared machine limit the conclusions.

| Matrix | Types | Threads | Pairs | SpCraft ms | CombBLAS ms | Ratio [95% CI] | CV %, SpCraft / CombBLAS |
|---|---|---:|---:|---:|---:|---|---|
| amazon0312.mtx squared | FP64/int32 | 1 | 11 | 1317.391 | 1451.554 | 0.908 [0.867, 0.922] | 2.8 / 1.5 |
| amazon0312.mtx squared | FP64/int32 | 4 | 15 | 575.001 | 415.151 | 1.385 [1.311, 1.435] | 4.8 / 3.9 |
| amazon0312.mtx squared | FP64/int32 | 8 | 15 | 338.489 | 231.536 | 1.462 [1.446, 1.629] | 7.1 / 2.0 |
| amazon0312.mtx squared | FP64/int32 | 16 | 15 | 230.559 | 135.358 | 1.703 [1.599, 1.862] | 3.1 / 7.5 |
| cant.mtx squared | FP64/int32 | 1 | 10 | 1649.860 | 1573.795 | 1.048 [1.045, 1.051] | 0.3 / 0.3 |
| cant.mtx squared | FP64/int32 | 4 | 15 | 576.357 | 420.408 | 1.371 [1.354, 1.391] | 1.2 / 1.4 |
| cant.mtx squared | FP64/int32 | 8 | 15 | 315.938 | 234.111 | 1.350 [1.340, 1.367] | 1.2 / 1.4 |
| cant.mtx squared | FP64/int32 | 16 | 15 | 188.511 | 225.237 | 0.837 [0.815, 0.966] | 12.7 / 10.7 |
| ecology1.mtx squared | FP64/int32 | 1 | 15 | 278.245 | 318.385 | 0.874 [0.861, 0.887] | 1.3 / 1.5 |
| ecology1.mtx squared | FP64/int32 | 4 | 15 | 724.313 | 138.534 | 5.228 [5.195, 5.284] | 1.5 / 2.4 |
| ecology1.mtx squared | FP64/int32 | 8 | 15 | 604.988 | 82.307 | 7.350 [7.236, 8.923] | 2.2 / 9.9 |
| ecology1.mtx squared | FP64/int32 | 16 | 15 | 502.003 | 57.247 | 8.769 [8.091, 9.699] | 4.0 / 8.1 |
| ER_V20000_D32 squared | FP64/int32 | 1 | 11 | 1465.189 | 1526.803 | 0.960 [0.957, 0.963] | 0.6 / 0.4 |
| ER_V20000_D32 squared | FP64/int32 | 4 | 15 | 418.229 | 415.714 | 1.006 [0.995, 1.011] | 1.0 / 0.8 |
| ER_V20000_D32 squared | FP64/int32 | 8 | 15 | 220.976 | 230.897 | 0.957 [0.947, 0.990] | 5.2 / 5.3 |
| ER_V20000_D32 squared | FP64/int32 | 16 | 15 | 126.104 | 138.740 | 0.909 [0.887, 0.961] | 3.4 / 4.6 |
| ER_V20000_D8 squared | FP64/int32 | 1 | 15 | 81.624 | 88.017 | 0.927 [0.899, 0.944] | 1.7 / 1.4 |
| ER_V20000_D8 squared | FP64/int32 | 4 | 15 | 36.260 | 28.217 | 1.285 [1.052, 1.338] | 10.2 / 7.4 |
| ER_V20000_D8 squared | FP64/int32 | 8 | 15 | 19.651 | 15.606 | 1.259 [1.235, 1.600] | 15.3 / 16.6 |
| ER_V20000_D8 squared | FP64/int32 | 16 | 15 | 12.237 | 10.139 | 1.207 [1.119, 1.522] | 6.0 / 21.0 |
| RMAT_S12_E8 squared | FP64/int32 | 1 | 15 | 186.529 | 190.019 | 0.982 [0.981, 0.983] | 0.1 / 0.1 |
| RMAT_S12_E8 squared | FP64/int32 | 4 | 15 | 54.177 | 76.867 | 0.705 [0.648, 0.747] | 3.9 / 5.5 |
| RMAT_S12_E8 squared | FP64/int32 | 8 | 15 | 28.863 | 49.279 | 0.586 [0.510, 0.672] | 10.0 / 9.3 |
| RMAT_S12_E8 squared | FP64/int32 | 16 | 15 | 17.401 | 32.256 | 0.539 [0.518, 0.604] | 11.4 / 9.7 |

Maximum verification error: 0.

Raw per-iteration samples and build/runtime metadata accompany this report.
