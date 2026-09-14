# OmpHashSpGEMM / CombBLAS comparison

[SpGEMMBenchmark.cpp](SpGEMMBenchmark.cpp) compares the sole SpCraft kernel,
`OmpHashSpGEMM`, against unmodified
[`LocalSpGEMMHash`](https://github.com/hongyx11/CombBLAS/blob/2381a1f7a028b9f56d8531c0b7b157ae2ab27572/include/CombBLAS/mtSpGEMM.h)
at commit `2381a1f7a028b9f56d8531c0b7b157ae2ab27572`.
Both compute A*B directly from identical canonical DCSC inputs and return
column/row-sorted arrays of `std::tuple<IT,IT,NT>`. SpCraft owns its raw array
through `CooMatrix::entries`. There is no output-format selector or alternate
SpCraft multiplication implementation.

The [source mapping](SourceMapping.md) documents each stage and the explicit
checked-arithmetic, thread-discovery and ownership adaptations. The last
[BigRed native timing report](../results/combblas/spgemm/MappedTimingResults.md)
measured this mapped algorithm before its public rename and the COO ownership
consolidation. Its original source hashes and raw observations are preserved;
that historical run is not a new measurement of the renamed build.

## Library API

```cpp
#include "SpCraft.h"
using Ring = spcraft::PlusTimesRing<double>;
// A and B are canonical DcscMatrix<int,double>, A.n == B.m.
auto C = spcraft::OmpHashSpGEMM<Ring>(A, B);
// C.entries[p] is (row, column, value); no intermediate compressed output.
auto csr = C.ToCsr(); // Explicit conversion only if needed by a consumer.
```

The implementation lives in [mtSpGEMM.h](../../include/kernel/mtSpGEMM.h).
The library kernel needs no CombBLAS or MPI dependency. IT, NT and OT retain
the library's type order. For a reference comparison, use OT=IT and matching NT.
Canonical inputs have sorted unique row IDs in each stored column and sorted
unique stored column IDs. `A.n+1` must fit IT and be at most 2^24; full work and
output counts must fit OT. CSC preparation uses `DcscMatrix::FromCsc`, which
copies storage but does not sort or merge duplicates. The
[example](../../examples/spcraft/e03_column_spgemm.cpp) shows this boundary.

## Build

From the repository root:

```sh
git clone https://github.com/hongyx11/CombBLAS.git build/_deps/combblas-src
git -C build/_deps/combblas-src checkout 2381a1f7a028b9f56d8531c0b7b157ae2ab27572
cmake -S build/_deps/combblas-src -B build_combblas \
  -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON -DBUILD_TESTING=OFF
cmake --build build_combblas --target CombBLAS --parallel 4
cmake -S . -B build_spgemm -DCMAKE_BUILD_TYPE=Release \
  -DSPCRAFT_BUILD_EXAMPLES=OFF -DBUILD_TESTING=ON \
  -DSPCRAFT_BUILD_BENCHMARKS=ON -DSPCRAFT_USE_COMBBLAS=ON \
  -DCombBLAS_DIR=build_combblas/CombBLAS \
  -DSPCRAFT_COMBBLAS_REVISION=2381a1f7a028b9f56d8531c0b7b157ae2ab27572
cmake --build build_spgemm --target spgemm_combblas_benchmark test_mtspgemm --parallel 4
OMP_NUM_THREADS=4 build_spgemm/tests/test_mtspgemm
```

`SPCRAFT_USE_COMBBLAS` defaults to OFF; reference comparison builds require MPI
and OpenMP. `test_mtspgemm` compares intermediate helpers, every output value
bit and the semiring operation/argument sequence against actual upstream code.
The default build exercises the same kernel with independent references.

## Run and timing scope

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores \
  OMP_WAIT_POLICY=active build_spgemm/benchmarks/combblas/spgemm_combblas_benchmark \
  --vertices 20000 --degree 8 --threads 1,4,16,32 \
  --iterations 20 --max-time 45 --output build_spgemm/native.json

python3 benchmarks/scripts/run_spgemm_comparison.py \
  --binary build_spgemm/benchmarks/combblas/spgemm_combblas_benchmark \
  --output-dir benchmarks/results/combblas/spgemm/tuples-local \
  --cases er8,er32,rmat,hypersparse --threads 1,4,16 \
  --iterations 20 --max-time 45
```

Use `--matrix path.mtx` for A and `--right-matrix path.mtx` for B; otherwise
compute A*A. Generated inputs support ER and R-MAT. `--column-stride 64`
embeds coordinates at 64-fold spacing for hypersparse shapes. The seven-case
suite includes ER degree 8/32, R-MAT, hypersparse, amazon0312, cant and ecology1.
Pass `--dataset-dir` to the runner to select the SuiteSparse dataset root.

The executable supports FP32/FP64 and matching int32/int64 index/offset types.
`--precision both --index both --offset both` selects all four combinations.
The suite runner uses FP64/int32. Full independent Eigen structure/value checks
precede timing at each thread count. Both clocks include lookup/setup, work
estimation, symbolic, prefixes, numeric, sorting, allocation and destruction.
Input preparation and output conversion are outside timing; neither native
path performs output conversion. Three warmup rounds precede alternating
AB/BA pairs. The default limits are 20 pairs or 45 seconds per configuration.

The report records all samples, CV and paired bootstrap 95% intervals.
Ratio = SpCraft median / CombBLAS median; below one favors SpCraft. Intervals
describe sampled rounds in one allocation, not variation across allocations.
`--no-plot` requires only Python's standard library; plotting needs Matplotlib.
Use `--analyse-only` to regenerate a native-pair report.

## BigRed CPU debug

From an isolated workspace with dependencies under `dependencies/`:

```sh
sbatch benchmarks/combblas/slurm/spgemm.sbatch
```

The script verifies the reference revision, builds and runs correctness tests,
and runs the native comparison in an exclusive 128-core CPU debug allocation.
`SPCRAFT_THREADS`, `SPCRAFT_ITERATIONS` and `SPCRAFT_MAX_TIME` override the sweep
and budget. Results go to `benchmarks/results/combblas/spgemm/tuples-bigred-JOBID`.
It does not swap production headers or select a different multiplication path.

## Historical diagnostics

The former CSR and general column drivers and profiler ablations are removed.
[CSR audit](CorrectnessAudit.md), [general column audit](ColumnCorrectnessAudit.md),
[general kernel mapping](GeneralKernelMapping.md) and [phase observations](PhaseProfiling.md)
are historical records, not instructions for an available alternative kernel.
Raw measurements under `benchmarks/results/combblas/spgemm/` remain unchanged.
The separate `spgemm_conversion_benchmark` measures external CombBLAS storage
compression only; it does not contain another SpCraft SpGEMM implementation.
