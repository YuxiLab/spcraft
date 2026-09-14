---
name: spcraft-review
description: Review SpCraft C++/CUDA changes against this repo's ownership, semiring, OpenMP, CUDA, and build conventions. Use when reviewing a diff, a branch, a PR, or a specific file in the SpCraft sparse library — especially changes touching include/core, include/kernel, src/cuSpMV.cu, tests, benchmarks, or the Python bindings.
---

# SpCraft code review

Review changed C++/CUDA code against the invariants below. These are the rules
this codebase actually holds itself to — a generic C++ review will miss most of
them.

## Scope

Default to the uncommitted diff (`git diff HEAD`). If the user names a branch,
PR, or path, review that instead. Read the full file around each change; these
are template-heavy headers where the bug is usually in the interaction between
a change and a distant `static_assert` or explicit instantiation.

Report findings most-severe first. For each: file:line, what breaks, and a
concrete failing input. Distinguish "this is wrong" from "this is unidiomatic
for the repo" — say which. If nothing is wrong, say so plainly rather than
padding with style nits.

## Ownership contract (include/core/)

`CooMatrix`, `CsrMatrix`, `CscMatrix`, and `DenseVector` all follow one pattern.
Any new container or any edit to these must keep it:

- Raw public pointers, `memowned` flag, move-only. Copy ctor and copy assignment
  are `= delete`; copying must go through explicit `Clone()`.
- The pointer constructor sets `memowned = false` — it is the view/borrow path,
  and is also how device (CUDA) memory enters these classes.
- Move ctor and move assignment must call `rhs.Reset()` after stealing. Move
  assignment must `SafeDelete` its own storage first and guard `this != &rhs`.
- `Allocate` must build the replacement *before* freeing the old storage, so a
  throwing allocation leaves the object intact.
- Every `SafeAllocate` frees all partial allocations before throwing
  `std::bad_alloc`.
- Storage is `std::calloc`/`std::free`, not `new`/`delete`. Do not mix.
- Signed index and offset types get `if constexpr (std::is_signed_v<...>)`
  negativity checks; unsigned instantiations must compile without them.

### Known traps in this pattern

- **`SafeDelete` takes pointers by value.** Its `row_ptr = nullptr;` lines null
  local copies only — they are dead stores. Callers must null or overwrite the
  members themselves. A new call site that relies on `SafeDelete` nulling the
  object's members is a use-after-free.
- **`Reset()` sets `memowned = false` and does not free.** It releases
  ownership. Calling it on a live owning object leaks. It is only correct on the
  moved-from source, or on a view.
- `Clone()` of an empty matrix must tolerate null members — the guards around
  `std::copy` are load-bearing.

## Semiring kernels (include/kernel/)

- Every kernel is templated on `SemiRing` and opens with
  `static_assert(std::is_same_v<typename SemiRing::ValueType, NT>, ...)`.
  A new kernel without it will fail confusingly deep in instantiation.
- Accumulators initialize to `SemiRing::kAdditiveIdentity`, never `NT{0}`, and
  combine via `SemiRing::Add` / `SemiRing::Multiply`, never `+` / `*`.
  A literal `0` or `+` in a kernel body is a bug for non-arithmetic rings.
- `PlusTimesRing<bool>` is the OR-AND ring used for BFS. Check that changes
  still behave under it — `bool` saturates, so `sum += x` and `sum = sum + x`
  are not interchangeable in the ways one might assume.
- Dimension mismatches throw `std::invalid_argument` from host kernels. CUDA
  entry points **return `cudaError_t` instead of throwing** — do not add throws
  to a `.cuh`/`.cu` path.
- Output nnz accumulation must be overflow-checked against
  `std::numeric_limits<OT>::max()` before narrowing to `OT` (see
  `spgemm_openmp`), throwing `std::overflow_error`.

## OpenMP

- Loop pragmas use `default(none)` and must name **every** referenced variable
  in `shared(...)`. Adding a variable to a parallel loop body without adding it
  to the clause is a compile error on GCC and a silent sharing bug elsewhere —
  check the clause on every touched pragma.
- Pragmas are guarded by `#ifdef _OPENMP` so serial builds still compile.
- Verify the schedule matches the work: `static` for uniform per-row cost,
  `dynamic` for irregular (SpGEMM row expansion).
- Per-thread scratch must be declared inside the loop body, not hoisted.

## CUDA (src/cuSpMV.cu, include/kernel/cuSpMV.cuh)

- The public entry point validates, then delegates to a
  `detail::...Impl<...>::Run` struct. Templates are **not** defined in the
  header — the definition lives in the `.cu` and is reached through explicit
  instantiation.
- New type or block-size combinations must be added to the
  `BOOST_PP_SEQ_FOR_EACH_PRODUCT` sequence in `src/cuSpMV.cu`. Using an
  uninstantiated combination is a link error, not a compile error — check that
  any newly-used `<IT, NT, OT, kBlockSize>` is in the product.
- Block size is `static_assert`ed to be a power of two in [1, 1024], and the
  kernel carries `__launch_bounds__(kBlockSize)`. The shared-memory tree
  reduction is only correct for power-of-two sizes.
- Matrices passed to CUDA kernels must wrap **device** memory. A host-allocated
  `CsrMatrix` reaching a kernel launch is a silent wrong-answer bug — check the
  provenance of every buffer at the call site.
- Null-pointer checks return `cudaErrorInvalidDevicePointer`; bad shapes return
  `cudaErrorInvalidValue`; `m == 0` returns `cudaSuccess` early.
- Launches are async on the given stream. Any correctness check after a launch
  needs a sync; benchmarks must sync before stopping the timer.

## Build and layout (AGENTS.md rules)

- **No absolute paths** anywhere — in source, CMake, or scripts. Pass paths as
  parameters. This is an explicit project rule.
- Binaries take arguments via `cxxopts`, not hand-rolled `argv` parsing.
- `main` files include only `"SpCraft.h"`, not individual core headers.
- Folder responsibilities: `examples/` demonstrates API usage, `benchmarks/`
  measures performance and compares against MKL/cuSPARSE/Ginkgo, `tests/`
  checks correctness. A perf harness in `tests/` or an assertion-free program in
  `tests/` is misfiled.
- Optional backends stay behind `SPCRAFT_USE_{MPI,CUDA,MKL}` in both CMake and
  `#ifdef`s. A new backend header must be included conditionally in
  `include/SpCraft.h`.
- New test executables need both `add_executable` and `add_test` in
  `tests/CMakeLists.txt`, guarded by the matching option when backend-specific.

## Python bindings (python/bindings.cpp)

- The boundary **copies** arrays deliberately, so a Python array can never
  invalidate or alias SpCraft storage. A change that hands out a pointer into
  C++-owned memory breaks the documented ownership guarantee.
- Returned NumPy arrays own their buffer via an `nb::capsule` deleter that
  matches the allocation (`delete[]` for `new[]`).
- Sizes narrowing into `Offset`/`Index` are range-checked before the cast.

## Tests

Tests are plain `main()` programs returning non-zero on failure, not a
framework. A new kernel should get: a hand-checked small matrix with a known
result, an empty-row case, a dimension-mismatch case asserting the throw, and
where an Eigen or MKL reference exists, a cross-check against it.
