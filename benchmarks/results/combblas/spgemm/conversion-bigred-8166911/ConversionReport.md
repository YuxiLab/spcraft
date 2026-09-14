# Upstream tuple compression comparison

Job 8166911 completed (0:0) in 56 seconds on exclusive CPU-debug node nid0640. Twelve alternating serial/parallel pairs were measured at each thread count. Every output offset, column ID, row index and value matched between constructors. Native multiplication, compression, and DCSC destruction were clocked separately; tuple destruction is excluded equally.

All times below are medians in milliseconds. These measurements are on a different node/allocation from the phase run. Native multiplication itself varies materially between variants, particularly at 128 threads; do not infer a corrected end-to-end total by adding these medians to another job's native time.

| Input | Threads | Constructor | Samples | Native ms | Conversion ms | DCSC free ms | Total ms |
|---|---:|---|---:|---:|---:|---:|---:|
| cant.mtx | 128 | parallel | 12 | 93.514 | 5.606 | 0.001 | 99.143 |
| cant.mtx | 128 | serial | 12 | 146.491 | 73.917 | 0.002 | 220.535 |
| cant.mtx | 16 | parallel | 12 | 128.696 | 25.972 | 0.002 | 154.675 |
| cant.mtx | 16 | serial | 12 | 131.796 | 64.562 | 0.002 | 196.394 |
| ecology1.mtx | 128 | parallel | 12 | 73.539 | 1.684 | 0.001 | 75.243 |
| ecology1.mtx | 128 | serial | 12 | 112.565 | 83.979 | 0.003 | 185.524 |
| ecology1.mtx | 16 | parallel | 12 | 53.195 | 20.754 | 0.001 | 73.987 |
| ecology1.mtx | 16 | serial | 12 | 55.073 | 48.724 | 0.002 | 103.777 |

The serial reference-taking constructor scans all tuples three times on the calling thread. The five-argument parallel constructor partitions tuple runs and copies rows/values with OpenMP, using two temporary arrays of length nnz. It substantially reduces the conversion region on these inputs; allocation/placement interactions with subsequent native calls remain visible in the raw data.
