# Sparse Craft Library

This is a teaching-oriented sparse computation library. It aims to provide a light-weight but clear-documented sparse kernel, sparse algorithm library with visualization and teaching tutorials.

At the ground layer, the library will focus on implementing the basic sparse matrix storage, sparse matrix kernel ( SpMV, SpMM, etc.) as well as reproducing some classic papers in the sparse computation community.

Please send email to `yuxihong@iu.edu` if you have any questions.

[Yuxi Hong](https://hongyx11.github.io/)

## Spack dependencies and installation

The [yuxi Spack repository](https://github.com/hongyx11/spack-packages) provides
the `yuxi.spcraft` recipe and `envs/spcraft-cpu` / `envs/spcraft-cuda` development
environments. They include CMake, Ninja, Eigen, Boost, fmt, fast_matrix_market,
cxxopts, GoogleTest, and Google Benchmark. CUDA, MPI, and MKL are optional recipe
variants; NVHPC is an optional compiler choice.

From a checkout of that repository, install the CPU development stack:

```bash
spack -e envs/spcraft-cpu install --test root
spack env activate envs/spcraft-cpu
```

To build an existing SpCraft checkout, register it before installation:

```bash
spack -e envs/spcraft-cpu develop --no-clone -p "$SPCRAFT_SOURCE" yuxi.spcraft@develop
spack -e envs/spcraft-cpu concretize --force
spack -e envs/spcraft-cpu install --test root
```

For CUDA, use `envs/spcraft-cuda` with a site configuration that selects the GPU
architecture and compatible compiler/toolkit versions. If a compatible toolkit
already exists, put its `nvcc` on `PATH` and register it first:

```bash
spack external find --scope user --not-buildable cuda
spack -C "$SPCRAFT_SITE_CONFIG" -e envs/spcraft-cuda install
```

Without an external registration, Spack installs the toolkit. An external
registered with `--not-buildable` must satisfy the package requirements; an
incompatible version produces a concretization error rather than a second
installation. Machine-specific choices belong in `config/<site>/` in the Spack
repository. Its existing `config/dell` is specific to the author's server.

Spack builds use `SPCRAFT_FETCH_DEPENDENCIES=OFF`: every dependency must already
be installed. Ordinary CMake source builds retain the default download fallback.
To install directly through CMake using installed dependencies:

```bash
cmake -S . -B build/install -DSPCRAFT_FETCH_DEPENDENCIES=OFF \
  -DSPCRAFT_INSTALL=ON -DBUILD_TESTING=OFF -DSPCRAFT_BUILD_EXAMPLES=OFF
cmake --build build/install --parallel
cmake --install build/install --prefix "$SPCRAFT_INSTALL_PREFIX"
```

Installed consumers use `find_package(SPCraft CONFIG REQUIRED)` and link
`SPCraft::SPCraft`. `SPCraft::SPCraftInput` provides the compiled input helpers;
CUDA-enabled installations also export `SPCraft::SPCraftCUDA`.

## Build on macOS with OpenMP

Install the Xcode Command Line Tools and the Homebrew dependencies:

```bash
brew bundle --file=Brewfile
```

The Brewfile includes Boost, Eigen, fmt, cxxopts, Google Benchmark, GoogleTest,
libomp, MPI, CMake, and Ninja. CMake discovers these installed packages before
using its download fallback. `fast_matrix_market` has no Homebrew core formula
and continues to use that fallback; MPI-enabled Caliper is installed separately
as described in the [Neovim workflow guide](docs/nvim-overseer.md).

The `macos-openmp` preset uses the system `cc`/`c++` drivers with libomp and MPI,
disables CUDA and MKL, and builds the examples, CPU benchmarks, and tests.

```bash
export PATH="$(brew --prefix open-mpi)/bin:$PATH"
export OpenMP_ROOT="$(brew --prefix libomp)"
cmake --preset macos-openmp
cmake --build --preset macos-openmp --parallel
ctest --preset macos-openmp
```

For a non-Homebrew installation, set `OpenMP_ROOT` to your libomp installation
prefix containing `include/omp.h` and `lib/libomp.dylib`. OpenMP is required by
this preset, and MPI compiler wrappers must be on `PATH`. Configuration fails
if either dependency cannot be found. Build output goes to
`build/macos-openmp`.

## Neovim profiling and experiments

See [the Neovim workflow guide](docs/nvim-overseer.md) for Oh My Overseer tasks
and Caliper profiling on the `yuxi/profiler` branch. The included `.profiler.json`
and **Local SUMMA smoke test** experiment run a small four-rank SUMMA input and
verify the product against Eigen, without downloading a dataset.
Use the **Local SUMMA profiling** experiment to select datasets, threads, and
MPI grids in Overseer, then press `P` to collect a report for each combination.
The same annotated sources now build ordinary examples, timing-only `_benchmark`
executables, and detailed `_profile` executables. The panel also selects warmups,
repetition/time limits, and measurement mode. See the
[standalone research draft](research/comment-profiler/main.tex) and its
[evaluation driver](research/comment-profiler/evaluate.py).

## CUDA code navigation on macOS

With the native `macos-openmp` build active, run
`python3 scripts/setup_macos_clangd.py --native-headers --check` to provide CUDA
host declarations while preserving the native MPI compilation database. For
Linux/CUDA implementation navigation, `python3 scripts/setup_macos_clangd.py`
prepares a separate parsing environment without a GPU driver or a library build. See
[the setup guide](scripts/clangd/README.md) for verification and editor details.

## Benchmark command-line options

Common measurement settings live in `benchmarks/common/BenchmarkOptions.h`:
`--iterations`, `--max-time`, `--output`, and `--text-report`. Backend input
settings live with the driver; the MKL benchmark uses `benchmarks/mkl/MKLOptions.h`
for matrix/generator selection, type choices, seed, and thread counts.

The MKL options implementation registers both groups with one parser, so
`bench_mkl_spmv --help` shows the complete command line. Existing flag names and
defaults are retained, including MKL's matching index/offset widths. Parsing
returns plain C++ settings; `cxxopts` is compiled in the option modules rather
than included by the MKL execution code. Shared input definitions remain
available to the other benchmark drivers.

```bash
build/bin/bench_mkl_spmv --matrix A.mtx --iterations 100 --max-time 2 --threads 1,4,8
```

The `benchmark_options.*` GoogleTest cases check the shared groups. With MKL
enabled, `mkl_options.*` also checks combined parsing, help, and backend constraints.

OpenMP and MKL performance drivers use Google Benchmark for warmup and timing.
Each configuration uses one driver thread and the requested kernel team size,
records wall and process CPU time, and retains SpCraft's text and JSON reports.
`--iterations` caps the measured invocations; `--max-time` sets the estimated
wall-time budget after warmup. Each invocation is one Google Benchmark repetition,
so JSON reports continue to contain individual timing samples.

Shared-memory correctness tests use GoogleTest, with each case discovered by
CTest. The shared entry point calls `TestRun`; GoogleTest accepts its standard
flags when a test binary is run directly:

```bash
ctest --test-dir build --output-on-failure
build/bin/test_csr_matrix --gtest_filter='*Clone*'
```

Google Benchmark and GoogleTest are linked only to their respective benchmark
and test targets. Disable `SPCRAFT_BUILD_BENCHMARKS` and `BUILD_TESTING` when
configuring library-only consumers.

## Private lecture notes

The optional `latex/` submodule contains private lecture notes and is not needed
to build or use SPCraft. Authorized collaborators can check it out with:

```bash
git submodule update --init latex
```


## Design Philosophy

A lot of class design of Sparse Craft Library is derived from [CombBLAS](https://github.com/PASSIONLab/CombBLAS).
Thank you Aydin!

## Sparse Matrix Storage

`CooMatrix<IT, NT, OT = IT>` stores one contiguous raw array:

```cpp
using COO = spcraft::CooMatrix<int, double>;
using Entry = COO::Entry;  // std::tuple<int, int, double>: row, column, value
COO matrix;
matrix.Allocate(2, 3, 4);
Entry* entries = matrix.entries;
entries[0] = {2, 1, 7.0};
entries[1] = {0, 3, -5.0};
COO view(entries, matrix.nnz, matrix.m, matrix.n); // borrows live entries
const auto csr = matrix.ToCsr();
```

COO owns allocation and entry lifetimes through `malloc/free`, explicit tuple
construction/destruction, and `memowned`. It is move-only; `Clone()` makes an
independent copy. Allocation builds the replacement before releasing old storage,
so failure preserves the owner. `Reset()` drops ownership without freeing and is
intended for moved-from objects or views. Empty matrices retain their shape.
The normal array allocation value-initializes every tuple component.

General COO permits arbitrary entry order and retains duplicates and explicit
zeros. `ToCsr()` groups by row while preserving within-row order and validates
coordinates. Matrix Market parsing uses temporary component vectors, then copies
into the owned `Entry*`; no vector backs the matrix. Python arrays are copied at
both boundaries, including when extracting a component of an entry array.

The native-output `OmpHashSpGEMM` returns this same COO owner directly, with
entries sorted by column then row. The former separate tuple container and
COO's three-array constructor/fields are removed: use `entries[p]` or structured
bindings instead of COO `row_id[p]`, `col_id[p]`, and `val[p]`. CSR, CSC and DCSC
retain their compressed arrays. See the [COO example](examples/spcraft/e01_CooMatrix.cpp).

## Sparse Kernels Implementation

`OmpHashSpGEMM<SemiRing>(A, B)` in [mtSpGEMM.h](include/kernel/mtSpGEMM.h) is the
single maintained OpenMP sparse-sparse multiplication implementation. It maps
CombBLAS's column lookup, full work estimation, symbolic hash counting, prefixes,
numeric hash accumulation and sorted tuple emission. No CombBLAS dependency is
needed to use the library kernel.

```cpp
using Ring = spcraft::PlusTimesRing<double>;
// A and B are canonical DcscMatrix<int, double>, with A.n == B.m.
auto C = spcraft::OmpHashSpGEMM<Ring>(A, B);  // DCSC -> CooMatrix
```

Inputs must have sorted unique stored column IDs and sorted unique row IDs in
each stored column. `A.n + 1` must fit IT and be at most 2^24, and full work and
output counts must fit OT. Output retains structural zeros. Convert CSC inputs
explicitly with `DcscMatrix::FromCsc`; that conversion preserves row order and
duplicates, so callers must supply canonical CSC. Convert the COO result with
`ToCsr()` when a consumer needs CSR.

The old CSR/CSC/DCSC compressed-output kernels and alternate tuple kernel are
removed. There are no compatibility entry points or hidden kernel-selection
flags. The [example](examples/spcraft/e03_column_spgemm.cpp),
[source correspondence](benchmarks/combblas/SourceMapping.md), and
[kernel tests](tests/kernel/test_mtSpGEMM.cpp) describe the supported contract.

Python's CSR `spgemm` method explicitly sorts and merges duplicate input
coordinates into canonical DCSC, calls this same kernel, then converts COO to
CSR. These boundary conversions cost additional time and are excluded from the
[native comparison benchmark](benchmarks/combblas/README.md).


## Optional Python interface

Passing `-DSPCRAFT_BUILD_PYTHON=ON` to cmake to activate python interface. Please use [uv](https://docs.astral.sh/uv/) to install the python environment.

to install `uv`, use
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then use `uv` to create a virutal environment and install the python interface.
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install the package
uv pip install -e .
```

Create COO or CSR matrices from NumPy-compatible arrays:

```python
import numpy as np
import spcraft

coo = spcraft.coo_matrix(
    row_indices=[0, 1, 1],
    column_indices=[1, 0, 2],
    values=np.array([2, 4, 3], dtype=np.float32),
    shape=(2, 3),
)
csr = coo.to_csr()

print(csr.row_offsets)
print(csr.column_indices)
print(csr.values)
```

Generate graph inputs directly with SPCraft's C++ implementations:

```python
er = spcraft.gen_er_graph(vertices=512, expected_degree=12, seed=2026)
rmat = spcraft.gen_rmat(scale=9, edge_factor=6, seed=2026)
```

Inputs and returned NumPy arrays are copied in this first implementation. This keeps ownership
unambiguous: modifying a Python array cannot invalidate or modify an SPCraft matrix.

## MPI sparse SUMMA

Build with `-DSPCRAFT_USE_MPI=ON`. `Mpi2DSummaSpGEMM<SemiRing>(A, B)` takes
`DistSpCSC` or `DistSpDCSC` inputs on the same `CommGrid2D`. The grid
owns its row and column communicators; distributed matrices share the grid
and own their native local storage. Each matrix stores global `m` and `n`;
the process grid determines its contiguous local dimensions. Construction does
not redistribute entries.

The synchronous algorithm broadcasts sparse inner-block panels along rows and
columns, calls local hash SpGEMM, and merges sorted COO partial products using
the semiring. Inner dimensions must match. The CSC implementation uses one
contiguous submatrix per rank, with dimensions derived from the chosen process
grid. It gathers each submatrix's `m`, `n`, and `nnz` along the process rows
and columns, then broadcasts matching A and B blocks at each stage. Roots use
borrowed views, and receivers allocate stage buffers. Both CSC and DCSC currently
require a square process grid; smaller edge submatrices are supported. The result is
`DistSpCOO`, with structural zeros retained and entries sorted by column then row.
Compiled kernels support `PlusTimesRing<float>` and `PlusTimesRing<double>` with
signed index/offset widths `(32,32)`, `(32,64)`, and `(64,64)`.

`CommGrid3D(world, layers)` creates `layers` square planes. Ranks are ordered
by layer, then row, then column; the communicator size must be
`layers * side * side`. Inner block `k` is stored only on layer `k % layers`.
`AOwner(i,k)` and `BOwner(k,j)` return the source rank responsible for each
input block. Each input uses `grid.Plane()` and the plane's full 2D packed
indices, leaving other layers' inner blocks empty. Inputs and shapes must agree
across participating ranks.

`Mpi3DSummaSpGEMM<SemiRing>(A, B, grid)` runs the assigned stages within each
plane and reduces sparse partial products along fibers using `SemiRing::Add`.
It returns `DistSpCOO<IT, NT, OT>` directly: layer zero receives the reduced
product and other layers receive a default-constructed empty matrix with no
grid ownership. Check `grid.Layer() == 0` to identify result owners, since a
valid product can also be empty. The reduced result shares its plane grid and
can outlive the 3D grid. Release all communicator owners before `MPI_Finalize`.
Calls are collective, with no communication/computation overlap; invoke them
from the MPI main thread when using `MPI_THREAD_FUNNELED`.

For a square Matrix Market input, these examples verify `A*A` against Eigen.
The 2D example takes the process grid dimensions and derives row and column block
extents from the matrix dimensions using ceiling division. A zero grid dimension
lets MPI choose it to match the participating rank count.

```bash
mpiexec -n 4 build/bin/e08_summa_2d --matrix A.mtx --grid-rows 2 --grid-cols 2
mpiexec -n 16 build/bin/e08_summa_2d --matrix A.mtx --grid-rows 4 --grid-cols 4
mpiexec -n 8 build/bin/e09_summa_3d --matrix A.mtx --layers 2
ctest --test-dir build -R 'test_mpi_(2dspgemm|3dspgemm|CommGrid3D)' --output-on-failure
```

`Bench_Mpi2DSummaSpGEMM` accepts Matrix Market files with `-a A.mtx -b B.mtx`
and computes `A*B` with one OpenMP thread per rank. A is required; omitting B
reuses A and computes `A*A`. Positional `A.mtx [B.mtx]` arguments also work.
The long option names are `--matrix-a` and `--matrix-b`.
Enable benchmarks in a Caliper build to generate repeated timing and
profiling variants using the `profiler.nvim` harness:

```bash
cmake --preset profiler -DSPCRAFT_BUILD_BENCHMARKS=ON \
  -DSPCRAFT_PROFILER_TOOLS=/path/to/profiler.nvim
cmake --build build/profiler --parallel 4 --target Bench_Mpi2DSummaSpGEMM_measurements
mpiexec -n 4 build/profiler/bin/Bench_Mpi2DSummaSpGEMM_benchmark \
  -a A.mtx -b B.mtx --profiler-warmup 1 --profiler-repeat 20
```

Use a square rank count. Input construction, output disposal, rank-entry barriers,
and the final Eigen correctness check are outside the measured call. Rectangular
inputs are supported when A's column count matches B's row count. Each rank
reads the input files and builds an Eigen reference. Rank 0 coordinates
stopping and prints the median of per-repetition maximum rank durations. The
`_profile` variant also records internal phases; see the
[profiling workflow](docs/nvim-overseer.md#repeated-benchmark-blocks) for trace
collection and runtime options. Defaults are one warmup and at most 1000
repetitions or 10 seconds. The ordinary target performs one correctness check
without timing and needs no Caliper dependency.
