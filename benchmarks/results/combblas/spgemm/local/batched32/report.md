# SpCraft / CombBLAS hash SpGEMM comparison

Each result is A². SpCraft computes CSR A×A; CombBLAS computes DCSC Aᵀ×Aᵀ with `LocalSpGEMMHash(..., sort=true)`. Input conversion and Eigen verification are outside timing. Each native call includes symbolic work, numeric work, sorting, allocation, and destruction. Native output formats differ: CSR versus tuples.

Ratio = SpCraft median / CombBLAS median; below 1 means SpCraft is faster. Intervals are 95% paired bootstrap intervals over alternating AB/BA samples. CV measures sample variability, not the performance gap. Few samples or a noisy shared machine limit the conclusions.

| Matrix | Types | Threads | Pairs | SpCraft ms | CombBLAS ms | Ratio [95% CI] | CV %, SpCraft / CombBLAS |
|---|---|---:|---:|---:|---:|---|---|
| amazon0312.mtx squared | FP64/int32 | 1 | 12 | 1274.682 | 1423.937 | 0.895 [0.883, 0.926] | 2.1 / 2.1 |
| amazon0312.mtx squared | FP64/int32 | 4 | 15 | 354.030 | 399.138 | 0.887 [0.864, 0.892] | 1.3 / 1.0 |
| amazon0312.mtx squared | FP64/int32 | 8 | 15 | 192.529 | 230.752 | 0.834 [0.813, 0.855] | 2.0 / 1.9 |
| amazon0312.mtx squared | FP64/int32 | 16 | 15 | 121.696 | 247.481 | 0.492 [0.461, 0.524] | 4.0 / 9.1 |
| cant.mtx squared | FP64/int32 | 1 | 10 | 1657.597 | 1587.617 | 1.044 [1.043, 1.066] | 2.0 / 2.9 |
| cant.mtx squared | FP64/int32 | 4 | 15 | 454.047 | 435.061 | 1.044 [1.040, 1.054] | 1.0 / 0.9 |
| cant.mtx squared | FP64/int32 | 8 | 15 | 237.755 | 237.709 | 1.000 [0.986, 1.012] | 1.3 / 1.7 |
| cant.mtx squared | FP64/int32 | 16 | 15 | 140.269 | 145.557 | 0.964 [0.902, 0.988] | 4.2 / 6.8 |
| ecology1.mtx squared | FP64/int32 | 1 | 15 | 254.898 | 316.101 | 0.806 [0.804, 0.809] | 0.4 / 0.5 |
| ecology1.mtx squared | FP64/int32 | 4 | 15 | 154.136 | 129.098 | 1.194 [1.173, 1.199] | 1.8 / 1.9 |
| ecology1.mtx squared | FP64/int32 | 8 | 15 | 95.437 | 83.291 | 1.146 [1.119, 1.266] | 2.6 / 9.6 |
| ecology1.mtx squared | FP64/int32 | 16 | 15 | 78.142 | 62.334 | 1.254 [1.165, 1.328] | 2.0 / 12.6 |
| ER_V20000_D32 squared | FP64/int32 | 1 | 11 | 1455.809 | 1514.727 | 0.961 [0.953, 0.970] | 0.5 / 1.2 |
| ER_V20000_D32 squared | FP64/int32 | 4 | 15 | 395.567 | 416.026 | 0.951 [0.939, 0.961] | 1.2 / 1.6 |
| ER_V20000_D32 squared | FP64/int32 | 8 | 15 | 217.637 | 233.935 | 0.930 [0.908, 0.960] | 1.7 / 3.6 |
| ER_V20000_D32 squared | FP64/int32 | 16 | 15 | 121.732 | 134.023 | 0.908 [0.900, 0.921] | 1.1 / 1.5 |
| ER_V20000_D8 squared | FP64/int32 | 1 | 15 | 82.561 | 91.298 | 0.904 [0.882, 0.941] | 2.3 / 1.7 |
| ER_V20000_D8 squared | FP64/int32 | 4 | 15 | 23.522 | 26.394 | 0.891 [0.809, 0.930] | 6.8 / 6.7 |
| ER_V20000_D8 squared | FP64/int32 | 8 | 15 | 13.134 | 14.045 | 0.935 [0.824, 0.980] | 8.3 / 12.0 |
| ER_V20000_D8 squared | FP64/int32 | 16 | 15 | 7.927 | 8.569 | 0.925 [0.907, 0.943] | 1.3 / 7.9 |
| RMAT_S12_E8 squared | FP64/int32 | 1 | 15 | 193.105 | 199.141 | 0.970 [0.967, 0.971] | 0.3 / 0.2 |
| RMAT_S12_E8 squared | FP64/int32 | 4 | 15 | 52.945 | 82.997 | 0.638 [0.634, 0.692] | 3.5 / 4.5 |
| RMAT_S12_E8 squared | FP64/int32 | 8 | 15 | 27.850 | 48.368 | 0.576 [0.550, 0.626] | 7.8 / 4.8 |
| RMAT_S12_E8 squared | FP64/int32 | 16 | 15 | 16.868 | 31.160 | 0.541 [0.533, 0.559] | 8.4 / 5.7 |

Maximum verification error: 0.

Raw per-iteration samples and build/runtime metadata accompany this report.
