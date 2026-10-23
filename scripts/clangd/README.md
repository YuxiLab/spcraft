# macOS clangd setup

## Native CMake builds

Select the `macos-openmp` configure preset in Antigravity or another CMake-aware
editor, then configure the project. The preset creates the relative link
`build/compile_commands.json -> macos-openmp/compile_commands.json`, matching
the database location in `.clangd`. Reconfiguration keeps it current as sources
are added. An existing database belonging to another build or parsing setup is
preserved with a warning instead of being overwritten.

For CUDA host declarations, add the headers needed by the editor's
`SPCRAFT_USE_CUDA=1` definition:

```sh
python3 scripts/setup_macos_clangd.py --native-headers --check
```

This downloads only the pinned CUDA runtime and internal compiler headers. It
adds a managed host-only include fragment to the existing `.clangd`,
preserving the active database and editor rules. It saves the original config in
`build/clangd-macos/clangd-before-native-headers.yaml`. Re-running setup reuses
verified downloads. Include paths are relative to the active compilation commands;
rerun after changing the build directory.
`--check` verifies the MPI CSC SUMMA source, `utils.h`, and `cuCsrMatrix.cuh`.
Restart the editor's clangd language server after changing the active database.
In Neovim, use `:LspRestart clangd`.

These headers support host declarations only; parsing `.cu` implementations needs
the Linux environment described below. Native CMake builds keep their configured
backends and do not link CUDA libraries.

## Linux/CUDA parsing environment

From the repository root, with Xcode Command Line Tools and Python 3.9+:

```sh
python3 scripts/setup_macos_clangd.py
```

Setup downloads checksum-pinned dependencies into the ignored
`build/clangd-macos/` directory and activates them through
`build/compile_commands.json`. It does not configure CMake, build binaries,
install system packages, or execute any downloaded Linux programs. An existing
unrelated compilation database is preserved; move it aside before activating this
one. Re-run setup after adding source files or moving the checkout. Repeated runs
reuse verified downloads.

The native Mac clangd targets Linux x86-64 for parsing and uses actual Linux/GCC
12 headers, CUDA 12.4.1 components, CCCL (CUB/Thrust), cuRAND, cuSPARSE, OpenMPI and
OpenMP headers. Header dependencies match the CMake fallback versions: Boost
Preprocessor 1.91.0, Eigen 5.0.1, fast_matrix_market 1.7.6, fmt 11.2.0 and cxxopts
3.3.1. Intel's headers-only `mkl-include` 2025.3.0 package supplies real oneMKL
declarations, with its license retained under `build/clangd-macos/mkl_include/`.
Download URLs and SHA-256 hashes are in `macos-deps.json`.

The database contains explicit entries for project sources and headers. `.cu`
files use CUDA mode and Clang's CUDA wrappers before GCC's C++ headers. `.cuh`
files hold host-side declarations only, so they use host C++ mode. CUDA declarations
remain enabled with `SPCRAFT_USE_CUDA=1`. Host C++ files enable OpenMP parsing;
CUDA files use host-only CUDA parsing, without enabling OpenMP. This does not
validate GPU code generation, linking, or execution.

MKL headers are available to all parsing commands. `SPCRAFT_USE_MKL=1` is enabled
for the MKL wrapper, tests, benchmark/options sources, and directly opened `SpCraft.h` and
`StablePch.h`; other targets keep their existing backend definitions. No MKL
libraries are installed or linked. This supports MKL navigation and completion
on Apple Silicon, but running MKL code still requires a supported build machine.

The checked-in `.clangd` takes compiler/toolkit/include paths from the database
instead of overriding them with machine-specific paths. Its existing host-header
language rules and diagnostic settings remain in effect.

## Editors

Restart Neovim to reload clangd after setup. Its existing clangd configuration
already attaches to C++ and CUDA buffers. No Neovim configuration change is needed.

In VS Code, use the `llvm-vs-code-extensions.vscode-clangd` extension and run
**clangd: Restart language server**. If needed, associate `*.cu` with `cuda-cpp`
and `*.cuh` with `cpp` in local editor settings. Other LSP editors use the same
database. This setup was checked with Apple clangd 17 from Xcode.

## Verification

```sh
python3 scripts/setup_macos_clangd.py --check
```

This checks 15 representative sources and headers and saves diagnostics under
`build/clangd-macos/*.check.log`. Every file is fully parsed; per-token editor
refactoring self-tests are restricted to line 1 to avoid unrelated Apple clangd
ExtractFunction failures. `--clangd PATH` selects the server used for these checks,
not the editor's server configuration. A nonzero exit means diagnostics were found;
setup remains active.

The MKL wrapper, test, and benchmark passed both Homebrew clangd 22.1.8 checks
and independent Clang syntax checks with MKL enabled. `SpCraft.h`,
`DenseVector-inl.h`, `CooMatrix-inl.h`, and `cuSpMV.cu` also passed clangd checks.
These checks parse declarations and template uses; they do not link or execute
MKL binaries.

At setup time, all three `src/cuda/*.cu` files, `test_cuspmv.cu`, the cuSPARSE
benchmark, `cuCsrMatrix.cuh`, `cuSpMV.cuh`, `SpCraft.h`, and `MatrixInput.cpp`
passed clangd and independent compiler syntax checks. Headers were also checked
by including them in a host translation unit.

Three existing tests have real API mismatches and therefore make `--check` exit
nonzero on this revision:

- `test_cucsr_matrix.cu` and `test_cupagerank.cu` call `cuCsrMatrix::FromHost`,
  but the class currently exposes a host-matrix constructor instead.
- `test_mtspmv.cpp` passes CSR to `OmpSpMV`, whose current signature takes CSC.

These diagnostics were reproduced by the compiler; setup does not suppress or
modify them. Optional CombBLAS and Python-binding dependencies are outside
this parsing environment and require their own headers and build definitions.

For a normal Linux build, restore the usual CMake-generated database link under
`build/`. The Mac database is for code navigation, not actual builds.
