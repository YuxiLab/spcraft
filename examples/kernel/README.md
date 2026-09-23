# Fast kernel iteration

Configure once, then keep the build directory between kernel edits:

```bash
cmake --preset cpu
cmake --build build --target ex02_spgemm
./build/bin/ex02_spgemm path/to/matrix.mtx
```

`cmake --build --preset examples -j 2` builds both kernel examples in `build/`.

These examples use `-O0` (`/Od` on MSVC). They verify correctness; use the
benchmark targets for performance measurements.

Three pieces of work are cached separately:

- `SPCraftInput` compiles cxxopts parsing and the Matrix Market reader once in
  `src/utils/MatrixInput.cpp`, with declarations in `include/utils/MatrixInput.h`.
- `../eigeninterface.cpp` compiles the Eigen reference operations once.
- `include/utils/StablePch.h` precompiles container and dependency headers. It deliberately
  excludes `SpCraft.h` and kernel headers, so kernel edits do not invalidate it.

The main programs still include `SpCraft.h` and instantiate the actual SpCraft
kernels. Editing a kernel recompiles the main and relinks the executable. The
input helper, Eigen helper, and precompiled header are reused. Editing container
headers, helper sources, or compiler options can require a larger rebuild.
The compiled helpers currently use `int64_t` indices/offsets and `double` values.

These utilities are available outside the examples. Include `SpCraft.h` to call
`spcraft::ParseMatrixInput` and `spcraft::ReadMatrixInput`, and link the optional
`SPCraftInput` target. The core `SPCraft` library remains header-only; programs
that do not use the compiled input utilities do not need to link them.

```cmake
target_link_libraries(my_program PRIVATE SPCraftInput)
target_precompile_headers(my_program PRIVATE <utils/StablePch.h>)
```

PCH is enabled per consumer rather than forced on every SpCraft user. The input
target inherits the selected build configuration; the example drivers retain
`-O0` for fast kernel iteration.

## Measured build times

Local measurements with GCC 12.3, Ninja, the CPU preset (MKL/OpenMP), and
`-O0 -g`, on 2026-09-23:

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
touch include/kernel/HashSpGEMMCsc_impl.h
/usr/bin/time -p cmake --build build --target ex02_spgemm
```

PCH is optional: configure with `-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON` to build
without it. Both examples were also built with PCH and OpenMP disabled to check
that they do not depend on headers being implicitly supplied by the cache.

The CSC SpGEMM implementation is currently incomplete. A nonempty input still
produces a verification failure and exit code 1; compilation caching does not
change that behavior.
