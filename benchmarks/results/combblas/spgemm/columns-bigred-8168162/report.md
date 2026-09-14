# Native tuple SpGEMM comparison

Primary comparison: identical DCSC inputs, sorted std::tuple<IT,IT,NT> arrays on both sides. Both clocks include lookup/setup, symbolic, numeric, sorting, initialized output allocation, and destruction. Neither native path converts the product. Input preparation and independent Eigen verification are excluded. The additional DCSC comparison uses the parallel CombBLAS compression constructor.

Five-backend order follows an odd-treatment Williams design (ten-round period). Ratios below one favor SpCraft. Intervals are paired bootstrap 95% intervals; they do not measure variation between node allocations. CV is sample variability, distinct from the implementation gap.

| Matrix | Types | Threads | Rounds | SpCraft tuples ms | Comb tuples ms | Tuple ratio [95% CI] | SpCraft DCSC ms | Comb DCSC ms | Tuple CV % (Sp / Comb) |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| amazon0312.mtx squared | FP64/int32 | 1 | 8 | 1091.230 | 1245.064 | 0.876 [0.875, 0.877] | 1095.437 | 1282.212 | 0.0 / 0.2 |
| amazon0312.mtx squared | FP64/int32 | 4 | 20 | 291.722 | 679.155 | 0.430 [0.429, 0.430] | 288.145 | 703.245 | 0.1 / 0.0 |
| amazon0312.mtx squared | FP64/int32 | 16 | 20 | 92.024 | 245.755 | 0.374 [0.373, 0.375] | 86.361 | 266.659 | 0.3 / 0.4 |
| amazon0312.mtx squared | FP64/int32 | 32 | 20 | 62.809 | 158.229 | 0.397 [0.387, 0.398] | 55.755 | 179.386 | 2.6 / 2.4 |
| amazon0312.mtx squared | FP64/int32 | 64 | 20 | 55.267 | 113.196 | 0.488 [0.428, 0.492] | 46.071 | 132.012 | 17.9 / 12.3 |
| amazon0312.mtx squared | FP64/int32 | 128 | 20 | 63.672 | 108.110 | 0.589 [0.446, 0.718] | 54.019 | 113.552 | 41.0 / 28.9 |
| cant.mtx squared | FP64/int32 | 1 | 6 | 1678.197 | 1617.033 | 1.038 [1.036, 1.039] | 1730.470 | 1659.171 | 0.1 / 0.1 |
| cant.mtx squared | FP64/int32 | 4 | 20 | 438.311 | 429.760 | 1.020 [1.019, 1.021] | 444.495 | 457.877 | 0.1 / 0.1 |
| cant.mtx squared | FP64/int32 | 16 | 20 | 128.839 | 128.167 | 1.005 [0.988, 1.008] | 123.792 | 152.542 | 0.2 / 1.7 |
| cant.mtx squared | FP64/int32 | 32 | 20 | 78.142 | 78.127 | 1.000 [0.910, 1.002] | 71.229 | 104.724 | 3.6 / 8.9 |
| cant.mtx squared | FP64/int32 | 64 | 20 | 59.641 | 59.849 | 0.997 [0.724, 1.033] | 51.085 | 85.819 | 17.4 / 29.3 |
| cant.mtx squared | FP64/int32 | 128 | 20 | 72.316 | 61.293 | 1.180 [0.679, 1.379] | 61.271 | 64.283 | 42.7 / 51.4 |
| ecology1.mtx squared | FP64/int32 | 1 | 20 | 261.325 | 372.740 | 0.701 [0.698, 0.705] | 271.331 | 407.028 | 0.7 / 0.4 |
| ecology1.mtx squared | FP64/int32 | 4 | 20 | 85.587 | 114.585 | 0.747 [0.745, 0.764] | 85.497 | 136.983 | 1.4 / 1.1 |
| ecology1.mtx squared | FP64/int32 | 16 | 20 | 47.241 | 48.509 | 0.974 [0.968, 0.978] | 40.670 | 68.858 | 1.0 / 1.0 |
| ecology1.mtx squared | FP64/int32 | 32 | 20 | 48.552 | 44.296 | 1.096 [0.968, 1.108] | 42.136 | 64.655 | 1.9 / 11.8 |
| ecology1.mtx squared | FP64/int32 | 64 | 20 | 52.685 | 43.280 | 1.217 [0.814, 1.233] | 47.249 | 57.746 | 20.7 / 35.8 |
| ecology1.mtx squared | FP64/int32 | 128 | 20 | 62.311 | 65.621 | 0.950 [0.678, 1.530] | 56.730 | 53.780 | 35.3 / 40.2 |
| ER_V20000_D32 squared | FP64/int32 | 1 | 7 | 1293.470 | 1352.102 | 0.957 [0.956, 0.958] | 1283.781 | 1400.739 | 0.1 / 0.1 |
| ER_V20000_D32 squared | FP64/int32 | 4 | 20 | 342.778 | 358.207 | 0.957 [0.956, 0.958] | 334.263 | 390.558 | 0.1 / 0.2 |
| ER_V20000_D32 squared | FP64/int32 | 16 | 20 | 107.223 | 110.213 | 0.973 [0.944, 0.975] | 96.891 | 139.284 | 2.2 / 2.6 |
| ER_V20000_D32 squared | FP64/int32 | 32 | 20 | 68.858 | 70.654 | 0.975 [0.877, 0.979] | 58.565 | 100.738 | 8.5 / 9.8 |
| ER_V20000_D32 squared | FP64/int32 | 64 | 20 | 53.778 | 57.138 | 0.941 [0.704, 0.958] | 42.943 | 88.472 | 25.6 / 26.8 |
| ER_V20000_D32 squared | FP64/int32 | 128 | 20 | 60.703 | 85.800 | 0.707 [0.459, 1.138] | 45.666 | 76.538 | 52.6 / 46.1 |
| ER_V20000_D8 squared | FP64/int32 | 1 | 20 | 75.304 | 82.598 | 0.912 [0.909, 0.913] | 74.568 | 86.118 | 0.3 / 0.2 |
| ER_V20000_D8 squared | FP64/int32 | 4 | 20 | 20.498 | 22.204 | 0.923 [0.919, 0.925] | 19.877 | 24.345 | 0.5 / 0.4 |
| ER_V20000_D8 squared | FP64/int32 | 16 | 20 | 6.724 | 8.271 | 0.813 [0.705, 1.001] | 6.071 | 7.610 | 18.4 / 16.0 |
| ER_V20000_D8 squared | FP64/int32 | 32 | 20 | 4.810 | 6.308 | 0.763 [0.618, 1.093] | 3.955 | 5.155 | 29.5 / 25.4 |
| ER_V20000_D8 squared | FP64/int32 | 64 | 20 | 4.726 | 6.322 | 0.748 [0.580, 1.168] | 3.197 | 4.755 | 34.4 / 28.4 |
| ER_V20000_D8 squared | FP64/int32 | 128 | 20 | 7.052 | 8.293 | 0.850 [0.753, 1.171] | 4.085 | 6.612 | 27.0 / 16.7 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 1 | 20 | 114.935 | 114.814 | 1.001 [0.999, 1.002] | 114.578 | 118.132 | 0.3 / 0.1 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 4 | 20 | 30.892 | 30.473 | 1.014 [1.008, 1.016] | 30.324 | 32.688 | 0.5 / 0.2 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 16 | 20 | 11.926 | 11.761 | 1.014 [0.806, 1.241] | 9.282 | 10.095 | 15.1 / 13.8 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 32 | 20 | 9.029 | 8.893 | 1.015 [0.736, 1.385] | 5.946 | 6.823 | 23.6 / 23.1 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 64 | 20 | 8.531 | 8.329 | 1.024 [0.735, 1.387] | 4.387 | 5.746 | 30.4 / 29.3 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 128 | 20 | 9.699 | 8.900 | 1.090 [0.979, 1.343] | 4.545 | 8.039 | 35.6 / 30.3 |
| RMAT_S12_E8 squared | FP64/int32 | 1 | 20 | 179.229 | 182.006 | 0.985 [0.984, 0.985] | 179.705 | 188.235 | 0.1 / 0.2 |
| RMAT_S12_E8 squared | FP64/int32 | 4 | 20 | 47.241 | 72.391 | 0.653 [0.651, 0.653] | 46.363 | 76.380 | 0.2 / 0.2 |
| RMAT_S12_E8 squared | FP64/int32 | 16 | 20 | 14.837 | 30.871 | 0.481 [0.446, 0.557] | 13.604 | 30.439 | 16.2 / 8.6 |
| RMAT_S12_E8 squared | FP64/int32 | 32 | 20 | 10.058 | 21.741 | 0.463 [0.408, 0.612] | 8.255 | 20.015 | 27.3 / 14.5 |
| RMAT_S12_E8 squared | FP64/int32 | 64 | 20 | 8.339 | 17.433 | 0.478 [0.404, 0.737] | 5.935 | 15.282 | 37.4 / 22.6 |
| RMAT_S12_E8 squared | FP64/int32 | 128 | 20 | 9.423 | 16.932 | 0.557 [0.445, 0.981] | 5.843 | 14.350 | 45.2 / 32.0 |

Maximum absolute verification error: 3.72529e-09.

Raw samples, source hashes and build/runtime metadata accompany this report.
