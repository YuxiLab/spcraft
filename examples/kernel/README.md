# Fast kernel iteration

Configure once, then keep the build directory between kernel edits:

```bash
cmake --preset cpu
cmake --build build --target ex02_spgemm
./build/bin/ex02_spgemm path/to/matrix.mtx
```

`cmake --build --preset examples -j 2` builds both kernel examples in `build/`.

When `SPCRAFT_BUILD_EXAMPLES=ON` and `SPCRAFT_BUILD_DOWNSTREAM=OFF`, these two
kernel examples remain available as explicit targets and are excluded from the
default build:

```bash
cmake --build build --target ex01_spmv ex02_spgemm --parallel
```

These examples and their Eigen reference helper follow the selected CMake build
type. Use `Release` or `RelWithDebInfo` for optimized execution on large matrices,
and `Debug` for unoptimized code. They verify correctness. Printed example timing
is not accurate; use `benchmarks/` for more accurate performance measurements.

Three pieces of work are cached separately:

- `SPCraftInput` compiles cxxopts option groups and the typed Matrix Market reader
  in `src/utils/MatrixInputOptions.cpp` and `src/utils/MatrixInput.cpp`.
- `src/eigen/eigeninterface.cpp` compiles the Eigen reference operations once
  as part of `SPCraft`; declarations live in `include/eigen/eigeninterface.h`
  and are available through `SpCraft.h`.
- `include/utils/StablePch.h` precompiles container and dependency headers. It deliberately
  excludes `SpCraft.h` and kernel headers, so kernel edits do not invalidate it.

Both main programs include `SpCraft.h`, parse the shared options, and dispatch to
`ExampleRun<IT, NT, OT>`. Each run loads its chosen storage types, calls the public
SpCraft kernel, verifies with Eigen, and reports the outcome. The shared Eigen
helper supports mixed index/offset widths through conversion to Eigen triplets.

Choose `--it int32|int64`, `--nt float|double`, and `--ot int32|int64`.
Defaults are `IT=int64`, `NT=double`, and `OT=IT`. The supported integer pairs are
`(32,32)`, `(32,64)`, and `(64,64)`; `IT=int64, OT=int32` is rejected.

```bash
build/bin/ex01_spmv matrix.mtx --it int32 --nt float --ot int64
build/bin/ex02_spgemm matrix.mtx --it int64 --nt double
```

SpMV accepts rectangular matrices. This SpGEMM example squares its input, so the
matrix must be square. Timings exclude loading and conversion and remain
illustrative; use the benchmark drivers for performance comparisons.

The compiled SpGEMM implementation lives in `src/kernel/HashSpGEMMCsc.cpp`.
Editing that implementation rebuilds the library and relinks its consumers.
Header, input-helper, Eigen-helper, and compiler-option changes can require a
larger rebuild.

These utilities are available outside the examples. Include `SpCraft.h` to call
`spcraft::ParseMatrixInput`, `spcraft::DispatchMatrixInputTypes`, and
`spcraft::ReadMatrixInput<IT, NT, OT>`, and link the optional `SPCraftInput` target.
The core `SPCraft` target is a shared library; the optional input target adds the
command-line parser and file-reader entry points.

To compose matrix flags with options owned by another main file, call
`spcraft::AddMatrixInputOptions(parser)`, register the main's other flags, parse
once, and call `spcraft::ReadMatrixInputOptions(arguments)`. The registration
helper adds `--matrix`, `--it`, `--nt`, and `--ot`; each main owns its help flag
and positional-argument policy. Link `cxxopts::cxxopts` when the main uses its
parser directly.

```cmake
target_link_libraries(my_program PRIVATE SPCraftInput)
target_precompile_headers(my_program PRIVATE <utils/StablePch.h>)
```

PCH is enabled per consumer rather than forced on every SpCraft user. The input
target, example drivers, and Eigen helper inherit the selected build
configuration.

## Measured build times

Historical measurements with GCC 12.3, Ninja, the CPU preset (MKL/OpenMP), and
the former forced `-O0 -g` configuration, on 2026-09-23. Optimized builds can take
longer to compile:

| Operation | Wall time |
| --- | ---: |
| Previous ex02 rebuild after a kernel edit | 13.25 s |
| Fresh build of both examples and dependencies, `-j 2` | 23.15 s |
| ex02 rebuild after editing CSC kernel code | 2.36 s |
| ex02 rebuild after touching `mtSpGEMM.h` | 2.41 s |
| Both examples after a CSC kernel edit, `-j 2` | 2.43 s |
| ex02 kernel edit after moving utilities into the library | 2.51 s |
| No-change ex02 build | 0.13 s |

The fresh build was measured in `build/cpu-fast-check`; incremental measurements
used the former `build/cpu` directory (the CPU preset now uses `build/`).
The fresh-build timing predates promoting input utilities to
`SPCraftInput`, which now follows the selected build configuration. The code-edit
check temporarily changed the kernel's diagnostic
and verified that the executable printed it, then restored the source. Neither
helper nor the PCH rebuilt during kernel-only changes. These are local timings
for the current kernel implementation, not a guarantee for larger future kernels.

To reproduce the incremental measurement without changing source contents:

```bash
touch src/kernel/HashSpGEMMCsc.cpp
/usr/bin/time -p cmake --build build --target ex02_spgemm
```

PCH is optional: configure with `-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON` to build
without it. Both examples were also built with PCH and OpenMP disabled to check
that they do not depend on headers being implicitly supplied by the cache.

The CSC SpGEMM implementation includes symbolic and numeric phases. The example
checks its result against Eigen and returns nonzero if verification fails.
