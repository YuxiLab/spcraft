# Coding Agent Guide For Sparse Craft Library

## SpCraft library Structure

SpCraft is a header-only C++20 template library. Almost everything lives in
`include/` and is compiled into whatever binary includes it; `src/` holds only
the CUDA translation units, which cannot be header-only.

- `include/core/` -- storage: `CooMatrix`, `CsrMatrix`, `CscMatrix`,
  `DenseVector`, and the device-side `cuCsrMatrix`.
- `include/kernel/` -- primitives that touch a matrix once: SpMV, SpMM, SpGEMM,
  SDDMM, and the dense-vector helpers in `mtBlas1.h`.
- `include/semiring/` -- the algebra the kernels are templated on.
- `include/graph/`, `include/linear_algebra/` -- algorithms built by calling a
  kernel repeatedly: PageRank, conjugate gradient, power iteration.
- `include/matrix_generator/`, `include/report/` -- inputs and benchmark output.

The file-name prefix says which backend a header implements: `mt` for
OpenMP-threaded host code, `cu` for CUDA, `mkl` for the oneMKL wrapper, no
prefix for backend-independent code. A backend-independent header holding the
option and result types sits beside the backends that share them
(`PageRank.h` beside `mtPageRank.h` and `cuPageRank.cuh`).

The library is heavily templated on three parameters, in this order: `IT` the
index type, `NT` the numerical value type, `OT` the offset type (defaulting to
`IT`). Keep that order and those names in new code.

## Formatting

`.clang-format` at the repository root is the single source of truth, and
`.clang-format-ignore` lists what it must not touch. Run
`clang-format -i` on every C++ file you add or change. CI fails on any
deviation, so an unformatted file will not merge.

`.clangd` configures the language server, including which flags header files
inherit. Two things there are load-bearing and easy to break:

- `.cuh` files are parsed as **host C++**, not CUDA. Clang supports CUDA up to
  12.8; a newer toolkit makes CUDA-mode parsing fail inside
  `__clang_cuda_runtime_wrapper.h` and leaves a broken preamble that degrades
  name lookup in ways that look like nonsense errors (`std::string` resolving
  to `int`). SpCraft's `.cuh` headers hold host-side declarations only, so they
  parse cleanly as C++.
- `-DSPCRAFT_USE_CUDA=1` is set globally. Headers have no entry in
  `compile_commands.json`, so clangd interpolates flags from whichever
  translation unit it judges nearest -- often a third-party dependency that
  never defines it, which greys out every `#ifdef SPCRAFT_USE_CUDA` block.

## General Coding Rules
- DO NOT use absolute path unless you ask the human and human agrees to do so, always try to pass
the path as input parameters.
- Prefer a named constant or a parameter to a magic number, and a comment that
  says *why* to one that repeats *what*.
- Comment the surprising part. A tricky index expression, a numerical
  cancellation, or a rule that only bites on one compiler deserves a sentence;
  a loop that copies an array does not.

## C++/CUDA Coding Rules

### OpenMP Pragma
use our own [omp wrapper](/home/exouser/code/iusparse/include/utils/omp/omp_wrapper.h) 
to build OpenMP pragma. Don't add things like `default(none)`, or `shared(A,x,y)` 
if `A`, `x`, `y` are arleady an array, as it will add my reading burden.  
Only add those critical primitives that if not added the code is not correct. 

### The container ownership contract

`CooMatrix`, `CsrMatrix`, `CscMatrix`, `DenseVector`, and `cuCsrMatrix` all
follow one pattern. Any new container, and any edit to these, must keep it:

- Raw public pointers, a `memowned` flag, and move-only semantics. The copy
  constructor and copy assignment are `= delete`; copying goes through an
  explicit `Clone()`.
- The pointer constructor sets `memowned = false`. It is the view/borrow path,
  and also how device memory enters these classes.
- The move constructor and move assignment call `rhs.Reset()` after stealing.
  Move assignment guards `this != &rhs` and frees its own storage first.
- `Allocate` builds the replacement **before** freeing the old storage, so a
  throwing allocation leaves the object intact and still owning something it
  can free.
- `SafeAllocate` computes every size **before** allocating anything, frees all
  partial allocations before throwing, and throws rather than returning a null
  pointer. Route element counts through `detail::CheckedElementCount<T>()` in
  `core/AllocationGuard.h`: `count * sizeof(T)` silently wraps otherwise, and
  the allocator hands back a buffer smaller than the caller believes.
- An empty container is representable. `nnz == 0` is a legitimate matrix with a
  shape, not an error -- rejecting it makes `Clone()` of an empty matrix and
  reading a structurally empty Matrix Market file throw.
- Host storage is `std::calloc`/`std::malloc` and `std::free`, never
  `new`/`delete`. Device storage is `cudaMalloc`/`cudaFree`. Do not mix.
- Signed index and offset types get `if constexpr (std::is_signed_v<...>)`
  negativity checks, so unsigned instantiations still compile.

Two traps in this pattern, both of which have bitten before:

- **`SafeDelete` takes its pointers by value.** Assigning `nullptr` to them
  inside it is a dead store. The caller must null or overwrite its own members.
- **`Reset()` releases ownership without freeing.** It is correct on a
  moved-from source or on a view, and leaks if called on a live owning object.

### Kernels

- Every kernel is templated on `SemiRing` and opens with
  `static_assert(std::is_same_v<typename SemiRing::ValueType, NT>, ...)`.
- Accumulators start at `SemiRing::kAdditiveIdentity`, never `NT{0}`, and
  combine through `SemiRing::Add` / `SemiRing::Multiply`, never `+` / `*`. A
  literal `0` or a bare `+` in a kernel body is a bug for non-arithmetic rings.
- `PlusTimesRing<bool>` is the OR-AND ring used for BFS. Check that changes
  still behave under it.
- Algorithms that divide or subtract -- PageRank, CG, power iteration -- need a
  field rather than a semiring. They are **not** templated on `SemiRing`; they
  fix `PlusTimesRing` for the multiply and `static_assert` a floating-point
  `NT`. Say so in the doc comment, so the omission reads as deliberate.
- Output nnz accumulation is overflow-checked against
  `std::numeric_limits<OT>::max()` before narrowing, throwing
  `std::overflow_error`.

### OpenMP
- Prefer concise OpenMP loop pragmas: use `#pragma omp parallel for` without `default(none)`.
  Rely on standard OpenMP data-sharing rules unless a variable requires non-default behavior.
  Add clauses such as `private`, `firstprivate`, `reduction`, or `schedule` only when required by the algorithm.
- Guard pragmas with `#ifdef _OPENMP` so serial builds still compile. The CI
  no-OpenMP job builds that path.
- Match the schedule to the work: `static` for uniform per-row cost, `dynamic`
  for irregular work such as SpGEMM row expansion.
- Declare per-thread scratch inside the loop body, never hoisted.

### CUDA

- Public entry points validate, then delegate to a `detail::...Impl<...>::Run`
  struct. Templates are **not** defined in the header: the definition lives in
  the `.cu` and is reached through explicit instantiation.
- New type or block-size combinations must be added to the
  `BOOST_PP_SEQ_FOR_EACH_PRODUCT` sequence in the matching `.cu`. An
  uninstantiated combination is a link error, not a compile error.
- CUDA entry points **return `cudaError_t`; they do not throw.** Bad shapes give
  `cudaErrorInvalidValue`, null buffers `cudaErrorInvalidDevicePointer`, and an
  empty problem returns `cudaSuccess` early. The exception is a container, whose
  constructor has no error code to return -- `cuCsrMatrix` throws, and says so.
- Block size is `static_assert`ed to be a power of two; the shared-memory tree
  reduction is only correct for those.
- Matrices reaching a kernel must wrap **device** memory. A host-allocated
  `CsrMatrix` passed to a kernel is a silent wrong answer, so check the
  provenance of every buffer at the call site. `cuCsrMatrix::View()` is the
  supported way to hand device buffers to a kernel that takes a `CsrMatrix`.
- Launches are asynchronous. Any correctness check after a launch needs a sync,
  and benchmarks must sync before stopping the timer.


## Examples, Benchmark, Tests

Main file should include only `SpCraft.h` for using the SpCraft implementation.
Examples folder should show users how to use the classes and the function in the SpCraft library.
Benchmarks folder should contain the performance test of SpCraft and comparison with other libraries.
Tests folder contains the tests that check the correctness of the classes and functions inside the
SpCraft library.

### Examples folder layout

Examples are teaching material. A student should be able to read one file top to
bottom and learn one thing.

- Group examples into topic subfolders under `examples/`. Two kinds exist:
  - Concept folders teach a prerequisite a student needs before reading SpCraft
    itself: `openmp/`, `mpi/`, `meta_programming/`.
  - Library folders show how to use SpCraft and its dependencies: `spcraft/`,
    `fast_matrix_market/`, `linear_algebra/`, `machine_learning/`.
- Each example is one self-contained `main` teaching a single concept. Name files
  `eNN_topic.cpp`, numbered in the order a student should read them, with the
  numbers unique inside a folder.
- Keep them short and heavily commented. Examples are the one place in this
  repository where explaining the obvious is correct.
- An example that uses SpCraft includes only `SpCraft.h`, and belongs in a library
  folder rather than a concept folder.
- Each subfolder owns a `CMakeLists.txt`, and `examples/CMakeLists.txt` must
  `add_subdirectory` it. A folder that is not added never builds and will silently
  rot.

### Benchmarks folder layout

- Give every library its own subfolder under `benchmarks/`, named after the library
  (`benchmarks/cusparse/`, `benchmarks/mkl/`, `benchmarks/ginkgo/`). SpCraft's own
  kernels go in `benchmarks/spcraft/`, named for the library under test rather than
  for the backend it uses.
- Inside a library folder, each main entry benchmarks exactly one kernel or algorithm.
  Do not measure SpMV and SpGEMM from the same binary; add another main instead.
- Put data structures and helpers shared by several main entries in a `.h` file rather
  than copying them between drivers.
- Name benchmark sources in CamelCase after the kernel they measure, adding the backend
  only when the folder does not already identify it: `cusparse/SpMVBenchmark.cu`, but
  `spcraft/SpMVOpenMPBenchmark.cpp` because SpCraft has more than one backend. Shared
  headers follow the same style (`BenchmarkCommon.h`). This matches the library headers
  (`CooMatrix.h`, `SemiRing.h`); only executable target names stay lowercase with
  underscores, since those are what a user types on the command line.
- Each library folder owns a `CMakeLists.txt` that finds its library and links it. The
  parent `benchmarks/CMakeLists.txt` adds the subdirectory behind the matching
  `SPCRAFT_USE_*` option, so a missing library never breaks the default build.

`benchmarks/scripts/run_spmv_pipeline.py` runs the whole comparison end to end:
confirm the dataset, verify every backend against the Eigen reference, time them,
and draw the figure. It refuses to benchmark a kernel whose result has drifted --
timing a wrong answer is worse than not timing at all. On a login node
(`sbatch` present, no `SLURM_JOB_ID`) it submits a batch job instead of running
here; `--executor local` forces it onto this machine and `--dry-run` writes the
sbatch script without submitting. Each stage runs alone via `--stage`.


### Tests folder layout

Tests are plain `main()` programs returning non-zero on failure, not a
framework. Keep them that way: no dependency beyond `SPCraft` and, where a
reference solve is useful, Eigen.

- **One executable per class or per kernel.** `test_csr_matrix.cpp` covers
  `CsrMatrix`, `test_cg.cpp` covers conjugate gradient. Do not append checks for
  a new class to an existing file.
- Count failures and keep going rather than returning on the first one, so a
  run reports everything that is broken at once. Print what failed on `stderr`
  and one success line on `stdout`.
- A new **container** gets: default construction, `Allocate` including the empty
  case, the pointer-constructor view, move construction, move assignment,
  self-move, `Clone` including the empty case, `Reset`, and a failed allocation.
  `test_csr_matrix.cpp` is the template.
- A new **kernel or algorithm** gets: a hand-checked small case with a known
  result, an edge case (empty row, dangling vertex, zero right-hand side), a
  dimension-mismatch case asserting the throw, and a cross-check against Eigen
  or another independent reference where one exists.
- **Check the rate, not only the tolerance.** An iterative method that converges
  is not necessarily correct; one that converges at the wrong rate has a bug the
  tolerance will not reveal. CG on a 2D Laplacian should take O(sqrt(kappa))
  iterations, and power iteration O(log(tol) / log|lambda2/lambda1|).
- Make failure output diagnosable: print the value you got and the value you
  expected, not just that they differ.
- Every new test needs both `add_executable` and `add_test` in
  `tests/CMakeLists.txt`, guarded by the matching `SPCRAFT_USE_*` option when it
  is backend-specific.

Prefer a deterministic failure to one that depends on the machine. Asking for an
allocation larger than RAM does not reliably fail -- overcommit satisfies it,
and AddressSanitizer aborts the process instead of throwing. Ask for one that
cannot be *sized* (`std::numeric_limits<Offset>::max()` elements) so the
overflow guard fires before the allocator is ever called.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request. Each job exists
because something can break in it that no other job would notice:

- **tests** -- GCC and Clang, `Debug` and `RelWithDebInfo`. `Debug` sets
  `-fsanitize=address`, so leaks and use-after-free fail the build. Clang is not
  redundant with GCC: it rejects `default(none)` pragmas that GCC accepts.
- **no-openmp** -- builds with OpenMP disabled, checking the `#ifdef _OPENMP`
  serial path still compiles.
- **cuda-build** -- compiles only, since no runner has a GPU. Still worth it: an
  explicit instantiation that is missing is a link error, which nothing else
  catches. It pins `CMAKE_CUDA_ARCHITECTURES` because `native` cannot detect a
  GPU that is not there.
- **python** -- builds the nanobind module and runs `pytest`, which the default
  CMake build never touches.
- **slides** -- builds every deck in `latex/beamer/`. The decks pull code out of
  `include/` and `src/` by line range, so a rename or an inserted line silently
  points a slide at the wrong code; a build failure here is the only automatic
  warning.
- **format** -- `clang-format --dry-run --Werror` over every tracked C++ file.

Run the equivalent locally before pushing:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSPCRAFT_BUILD_EXAMPLES=ON
cmake --build build -j && (cd build && ctest --output-on-failure)
clang-format --dry-run --Werror $(git ls-files '*.h' '*.cpp' '*.cu' '*.cuh')
```

On a machine with a GPU, `native` may not resolve correctly -- pass
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` explicitly if CUDA binaries fail at startup
with a misleading error such as `cudaMalloc ... out of memory` on an idle
device.

## Main File Input parameters
We use `cxxopts` to pass the parameters into the binaries. 

## Python Interface

Built only with `-DSPCRAFT_BUILD_PYTHON=ON`, through nanobind. `uv` manages the
environment; `uv sync --all-extras` builds the module and `uv run pytest
python/tests` exercises it. A rebuild after editing `python/bindings.cpp` needs
`uv sync --reinstall-package spcraft`, since an unchanged version number leaves
the cached wheel in place.

The layout is deliberate: `python/bindings.cpp` is the thin C++ boundary and
`python/spcraft/__init__.py` is where the friendly signatures, dtype dispatch,
and SciPy interoperation live. Put argument massaging in Python, not in C++.

Rules for the boundary:

- **Copy at the boundary, always.** A Python array must never alias SpCraft
  storage, in either direction. Handing out a pointer into C++-owned memory
  breaks the ownership guarantee the moment either side reallocates.
- A NumPy array returned to Python owns its buffer through an `nb::capsule`
  whose deleter matches the allocation -- `delete[]` for `new[]`. Use
  `AdoptAsNumpy` to transfer a freshly allocated buffer, `CopyToNumpy` to expose
  a copy of memory that C++ keeps owning.
- Wrap borrowed buffers in a `DenseVector` through the pointer constructor, so
  `memowned` is false and SpCraft never frees NumPy's memory or a capsule's.
- Range-check every size before narrowing it into `Index` or `Offset`.
- An algorithm returns its numerical result as an array and its metadata as a
  dict: `ranks, info = matrix.pagerank()`, following SciPy's `(x, info)`
  convention. Do not encode convergence in a sentinel value.
- Both `float32` and `float64` are instantiated. A new binding provides both.
- Every new binding gets a test in `python/tests/`, checked against NumPy,
  SciPy, or a closed form -- not against SpCraft's own C++ output, which would
  only prove the two agree with each other.
