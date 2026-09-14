# CSC and DCSC hash SpGEMM comparison

All four paths compute A*B directly by column expansion. The primary ratio compares SpCraft DCSC input/output with CombBLAS DCSC input/output. CombBLAS first produces sorted tuples; its DCSC timing includes compression and tuple destruction. The native tuple time is reported separately. SpCraft CSC is a storage-format diagnostic.

All times include symbolic, numeric, sorting, allocation, and destruction. Input preparation and full Eigen verification are outside timing. Four-backend measurement order follows a balanced Williams design. Ratios below one favor SpCraft; intervals are 95% paired bootstrap intervals over sampled rounds. CV measures within-run timing variability, not the implementation gap. These intervals do not account for variation between allocations.

| Matrix | Types | Threads | Rounds | CSC ms | DCSC ms | Comb tuples ms | Comb DCSC ms | DCSC ratio [95% CI] | DCSC / native tuples |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| amazon0312.mtx squared | FP64/int32 | 1 | 9 | 1121.670 | 1409.338 | 1263.016 | 1316.850 | 1.070 [1.069, 1.071] | 1.116 |
| amazon0312.mtx squared | FP64/int32 | 4 | 20 | 293.455 | 371.822 | 692.495 | 746.622 | 0.498 [0.497, 0.499] | 0.537 |
| amazon0312.mtx squared | FP64/int32 | 16 | 20 | 86.411 | 107.769 | 248.667 | 303.681 | 0.355 [0.354, 0.355] | 0.433 |
| amazon0312.mtx squared | FP64/int32 | 32 | 20 | 56.087 | 70.211 | 158.986 | 215.066 | 0.326 [0.326, 0.327] | 0.442 |
| amazon0312.mtx squared | FP64/int32 | 64 | 20 | 47.370 | 59.794 | 111.728 | 170.444 | 0.351 [0.350, 0.353] | 0.535 |
| amazon0312.mtx squared | FP64/int32 | 128 | 20 | 53.447 | 67.651 | 91.513 | 155.914 | 0.434 [0.431, 0.442] | 0.739 |
| cant.mtx squared | FP64/int32 | 1 | 7 | 1754.114 | 1744.421 | 1623.681 | 1684.179 | 1.036 [1.034, 1.036] | 1.074 |
| cant.mtx squared | FP64/int32 | 4 | 20 | 453.505 | 451.695 | 433.088 | 496.917 | 0.909 [0.908, 0.910] | 1.043 |
| cant.mtx squared | FP64/int32 | 16 | 20 | 126.431 | 125.594 | 128.404 | 193.493 | 0.649 [0.649, 0.650] | 0.978 |
| cant.mtx squared | FP64/int32 | 32 | 20 | 72.867 | 72.883 | 78.310 | 145.726 | 0.500 [0.498, 0.501] | 0.931 |
| cant.mtx squared | FP64/int32 | 64 | 20 | 51.014 | 52.693 | 58.698 | 131.171 | 0.402 [0.396, 0.406] | 0.898 |
| cant.mtx squared | FP64/int32 | 128 | 20 | 55.959 | 59.000 | 48.688 | 127.412 | 0.463 [0.448, 0.481] | 1.212 |
| ecology1.mtx squared | FP64/int32 | 1 | 20 | 242.969 | 372.143 | 383.360 | 428.610 | 0.868 [0.866, 0.871] | 0.971 |
| ecology1.mtx squared | FP64/int32 | 4 | 20 | 77.876 | 122.508 | 122.728 | 170.758 | 0.717 [0.715, 0.724] | 0.998 |
| ecology1.mtx squared | FP64/int32 | 16 | 20 | 39.419 | 59.713 | 48.139 | 97.162 | 0.615 [0.612, 0.616] | 1.240 |
| ecology1.mtx squared | FP64/int32 | 32 | 20 | 42.266 | 61.331 | 42.614 | 93.710 | 0.654 [0.652, 0.657] | 1.439 |
| ecology1.mtx squared | FP64/int32 | 64 | 20 | 48.091 | 66.269 | 39.446 | 95.303 | 0.695 [0.689, 0.703] | 1.680 |
| ecology1.mtx squared | FP64/int32 | 128 | 20 | 56.276 | 74.308 | 39.508 | 119.337 | 0.623 [0.598, 0.639] | 1.881 |
| ER_V20000_D32 squared | FP64/int32 | 1 | 9 | 1280.571 | 1310.192 | 1362.257 | 1432.257 | 0.915 [0.914, 0.916] | 0.962 |
| ER_V20000_D32 squared | FP64/int32 | 4 | 20 | 332.626 | 339.603 | 360.318 | 432.040 | 0.786 [0.785, 0.787] | 0.943 |
| ER_V20000_D32 squared | FP64/int32 | 16 | 20 | 96.617 | 98.451 | 110.397 | 183.884 | 0.535 [0.535, 0.536] | 0.892 |
| ER_V20000_D32 squared | FP64/int32 | 32 | 20 | 58.083 | 59.342 | 69.820 | 145.943 | 0.407 [0.406, 0.408] | 0.850 |
| ER_V20000_D32 squared | FP64/int32 | 64 | 20 | 41.967 | 42.709 | 52.473 | 133.480 | 0.320 [0.319, 0.321] | 0.814 |
| ER_V20000_D32 squared | FP64/int32 | 128 | 20 | 44.045 | 45.025 | 50.788 | 143.873 | 0.313 [0.298, 0.317] | 0.887 |
| ER_V20000_D8 squared | FP64/int32 | 1 | 20 | 75.845 | 82.289 | 84.957 | 91.346 | 0.901 [0.900, 0.902] | 0.969 |
| ER_V20000_D8 squared | FP64/int32 | 4 | 20 | 20.295 | 22.260 | 23.055 | 29.725 | 0.749 [0.747, 0.751] | 0.966 |
| ER_V20000_D8 squared | FP64/int32 | 16 | 20 | 6.202 | 6.831 | 7.744 | 13.917 | 0.491 [0.484, 0.502] | 0.882 |
| ER_V20000_D8 squared | FP64/int32 | 32 | 20 | 3.932 | 4.369 | 5.462 | 11.916 | 0.367 [0.362, 0.377] | 0.800 |
| ER_V20000_D8 squared | FP64/int32 | 64 | 20 | 3.025 | 3.357 | 4.682 | 11.810 | 0.284 [0.280, 0.299] | 0.717 |
| ER_V20000_D8 squared | FP64/int32 | 128 | 20 | 3.472 | 3.768 | 5.599 | 13.733 | 0.274 [0.271, 0.292] | 0.673 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 1 | 20 | 133.126 | 117.481 | 117.709 | 124.209 | 0.946 [0.944, 0.947] | 0.998 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 4 | 20 | 43.231 | 31.556 | 31.442 | 38.164 | 0.827 [0.822, 0.832] | 1.004 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 16 | 20 | 23.110 | 9.513 | 10.016 | 19.127 | 0.497 [0.446, 0.564] | 0.950 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 32 | 20 | 19.902 | 5.828 | 6.590 | 17.112 | 0.341 [0.303, 0.424] | 0.884 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 64 | 20 | 19.486 | 4.074 | 5.405 | 16.789 | 0.243 [0.214, 0.315] | 0.754 |
| ER_V20000_D8 squared stride64 | FP64/int32 | 128 | 20 | 22.515 | 4.064 | 6.257 | 18.817 | 0.216 [0.196, 0.287] | 0.650 |
| RMAT_S12_E8 squared | FP64/int32 | 1 | 20 | 177.892 | 180.313 | 184.331 | 197.261 | 0.914 [0.913, 0.915] | 0.978 |
| RMAT_S12_E8 squared | FP64/int32 | 4 | 20 | 46.747 | 47.392 | 73.681 | 86.757 | 0.546 [0.545, 0.547] | 0.643 |
| RMAT_S12_E8 squared | FP64/int32 | 16 | 20 | 13.789 | 13.450 | 29.659 | 41.714 | 0.322 [0.315, 0.332] | 0.453 |
| RMAT_S12_E8 squared | FP64/int32 | 32 | 20 | 8.575 | 8.054 | 19.622 | 31.519 | 0.256 [0.244, 0.268] | 0.410 |
| RMAT_S12_E8 squared | FP64/int32 | 64 | 20 | 6.118 | 5.910 | 14.234 | 26.806 | 0.220 [0.206, 0.230] | 0.415 |
| RMAT_S12_E8 squared | FP64/int32 | 128 | 20 | 5.495 | 5.210 | 11.469 | 25.177 | 0.207 [0.201, 0.222] | 0.454 |

## Sample variability

| Matrix | Threads | CSC CV % | DCSC CV % | Comb tuples CV % | Comb DCSC CV % |
|---|---:|---:|---:|---:|---:|
| amazon0312.mtx squared | 1 | 0.1 | 0.1 | 0.0 | 0.0 |
| amazon0312.mtx squared | 4 | 0.4 | 0.3 | 0.0 | 0.0 |
| amazon0312.mtx squared | 16 | 0.3 | 0.2 | 0.1 | 0.2 |
| amazon0312.mtx squared | 32 | 0.6 | 0.5 | 0.4 | 0.1 |
| amazon0312.mtx squared | 64 | 2.3 | 1.6 | 1.7 | 0.5 |
| amazon0312.mtx squared | 128 | 5.5 | 3.7 | 5.0 | 1.2 |
| cant.mtx squared | 1 | 0.3 | 0.0 | 0.0 | 0.1 |
| cant.mtx squared | 4 | 0.1 | 0.4 | 0.1 | 0.2 |
| cant.mtx squared | 16 | 0.2 | 0.2 | 0.5 | 0.2 |
| cant.mtx squared | 32 | 0.9 | 0.6 | 1.4 | 0.3 |
| cant.mtx squared | 64 | 3.4 | 3.3 | 4.3 | 0.7 |
| cant.mtx squared | 128 | 9.3 | 8.8 | 10.9 | 1.4 |
| ecology1.mtx squared | 1 | 4.4 | 2.2 | 2.7 | 2.5 |
| ecology1.mtx squared | 4 | 1.6 | 0.9 | 2.2 | 1.6 |
| ecology1.mtx squared | 16 | 1.1 | 0.7 | 0.9 | 0.4 |
| ecology1.mtx squared | 32 | 2.4 | 1.8 | 2.8 | 0.4 |
| ecology1.mtx squared | 64 | 6.5 | 4.0 | 9.4 | 1.9 |
| ecology1.mtx squared | 128 | 9.1 | 5.0 | 16.5 | 8.1 |
| ER_V20000_D32 squared | 1 | 0.1 | 0.1 | 0.1 | 0.1 |
| ER_V20000_D32 squared | 4 | 0.1 | 0.1 | 0.2 | 0.2 |
| ER_V20000_D32 squared | 16 | 0.4 | 0.3 | 0.4 | 0.2 |
| ER_V20000_D32 squared | 32 | 1.7 | 1.5 | 1.5 | 0.4 |
| ER_V20000_D32 squared | 64 | 6.0 | 5.4 | 6.3 | 1.2 |
| ER_V20000_D32 squared | 128 | 13.6 | 11.5 | 18.1 | 3.7 |
| ER_V20000_D8 squared | 1 | 0.2 | 0.2 | 0.1 | 0.3 |
| ER_V20000_D8 squared | 4 | 0.3 | 0.4 | 0.2 | 1.0 |
| ER_V20000_D8 squared | 16 | 1.3 | 1.3 | 2.0 | 2.9 |
| ER_V20000_D8 squared | 32 | 3.9 | 3.0 | 3.4 | 3.9 |
| ER_V20000_D8 squared | 64 | 9.8 | 7.9 | 12.4 | 6.6 |
| ER_V20000_D8 squared | 128 | 12.7 | 10.0 | 21.6 | 8.9 |
| ER_V20000_D8 squared stride64 | 1 | 0.2 | 0.1 | 0.2 | 0.5 |
| ER_V20000_D8 squared stride64 | 4 | 0.5 | 0.5 | 0.4 | 0.8 |
| ER_V20000_D8 squared stride64 | 16 | 1.2 | 5.7 | 17.9 | 12.6 |
| ER_V20000_D8 squared stride64 | 32 | 1.8 | 10.1 | 31.9 | 17.5 |
| ER_V20000_D8 squared stride64 | 64 | 2.5 | 15.6 | 42.5 | 20.9 |
| ER_V20000_D8 squared stride64 | 128 | 3.7 | 18.8 | 42.0 | 21.7 |
| RMAT_S12_E8 squared | 1 | 0.5 | 0.3 | 0.2 | 0.2 |
| RMAT_S12_E8 squared | 4 | 0.2 | 0.2 | 0.3 | 0.2 |
| RMAT_S12_E8 squared | 16 | 3.2 | 3.5 | 0.5 | 1.4 |
| RMAT_S12_E8 squared | 32 | 5.4 | 5.5 | 1.2 | 2.5 |
| RMAT_S12_E8 squared | 64 | 6.7 | 5.8 | 3.7 | 3.8 |
| RMAT_S12_E8 squared | 128 | 9.4 | 13.8 | 11.6 | 5.3 |

Maximum absolute verification error: 3.72529e-09.

Raw samples and build/runtime metadata accompany this report.
