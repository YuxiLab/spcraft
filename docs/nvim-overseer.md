# Neovim build and run workflows

The repository provides `.vscode/tasks.json`, which Overseer discovers without
loading project Lua code. It demonstrates sequential CMake actions and error
navigation. The personal configuration uses `hongyx11/oh-my-overseer.nvim`,
which also manages local and remote experiments. Run
`:OverseerRun` from the repository or one of its subdirectories.

## Actions

| Task | What it does |
| --- | --- |
| Configure | Configure the platform's CMake preset |
| Build | Build `test_csr_matrix` |
| Run | Run `test_csr_matrix` and show its output |
| Build and run | Build, then run after a successful build |
| Configure, build and run | Configure, then build, then run; stop on failure |
| Build all | Build all enabled targets |
| Test | Run CTest and show failure output |
| Build and test | Build all, then run CTest after a successful build |

macOS uses `macos-openmp`; Linux uses `openmp-examples`. On macOS, provide
`OpenMP_ROOT` as documented in the main README. The accompanying personal Neovim
configuration discovers libomp through Homebrew when this variable is absent.
The Test action expects the build to be configured with `BUILD_TESTING=ON`
(the default for a fresh configuration).

Both development presets enable `SPCRAFT_BUILD_DOWNSTREAM`, so the test and
benchmark targets are available as well as the kernel examples.

## Personal Neovim shortcuts

The installed NvChad configuration uses Space as its leader key:

| Shortcut | Action |
| --- | --- |
| Space cC | Configure, build and run |
| Space cR | Build and run |
| Space cb | Build |
| Space cm | Configure |
| Space cA | Build all |
| Space cT | Build and test |
| Space cr | Repeat the last completed workflow |
| Space cD | Manage experiments |
| Space cE | Open the active experiment |
| Space cP | Choose a CMake preset |
| Space cp | Pick any task, including Run and Test |
| Space cO / Space c0 | Toggle the task/output panel (capital O or zero) |
| Space co / Space cq | Open / close quickfix |
| `]q` / `[q` | Next / previous error |

Build shortcuts save modified files before starting. In the task panel, `o`
opens the selected task's output, and Enter opens actions such as restart or stop.
Failed process tasks open quickfix and their full output. Compiler messages with
source locations become navigable errors and editor diagnostics. A nonzero
program exit stops a workflow even if its output has no source location.

## Profile 2D SUMMA

Use the `yuxi/profiler` branch. The repository's `.profiler.json` configures
**Space p → Build and run profiler** (or `:ProfilerRun`). It configures and builds
`e08_summa_2d` in `build/profiler`, then runs four MPI ranks on the included
`examples/spcraft/inputs/summa_2d.mtx` with a 2×2 process grid. The example checks
the product against Eigen. This small input checks the workflow; use a larger
matrix for performance analysis by editing the `--matrix` argument.

Install MPI and MPI-enabled Caliper once. On macOS:

```bash
brew bundle --file=Brewfile
export PATH="$(brew --prefix open-mpi)/bin:$PATH"
export OpenMP_ROOT="$(brew --prefix libomp)"
git clone --branch v2.13.1 --depth 1 https://github.com/LLNL/Caliper.git build/profiler-caliper-src
cmake -S build/profiler-caliper-src -B build/profiler-caliper-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/profiler-caliper-install" \
  -DPython_EXECUTABLE="$(xcrun --find python3)" \
  -DWITH_MPI=ON -DWITH_FORTRAN=OFF -DBUILD_TESTING=OFF
cmake --build build/profiler-caliper-build --parallel 4
cmake --install build/profiler-caliper-build
```

On Linux use your installed MPI and Python rather than the Homebrew and Xcode
commands. Caliper and SpCraft must use the same MPI installation. The Xcode
Python selection above avoids Caliper 2.13.1's wrapper-generator incompatibility
with Python 3.15. An existing Caliper installation can instead be selected with
`caliper_DIR` on the first `cmake --preset profiler` configuration; set
`cali_query` in `.profiler.json` to its `bin/cali-query` executable.

The personal Neovim configuration discovers Homebrew libomp and MPI, loads
`.profiler.json` from the Git root even inside nested examples, and finds
`cali-query` in `build/profiler-caliper-install/bin`. Run `:checkhealth profiler`
to inspect the editor setup. The profiler loads Oh My Overseer before submitting
tasks, so profiling works immediately after opening Neovim.

Profiler commands insert only paired comments into the original C++ file:

```cpp
// @profiler-region broadcast start
MPI_Bcast(...);
// @profiler-region broadcast end
```

Inside a loop, `// @profiler-iter q start` and its matching `end` surround the
per-iteration work. Region pairs go inside the iteration pair. You can also write
markers by hand, around complete statements in one braced block. Markers must
nest correctly and remain within the same preprocessor branch.

The profiling build runs a grammar checker before generation:
`profiler_check` → `profiler_generate` → compilation. To check comments alone:

```bash
cmake --build --preset profiler --target profiler_check
```

The checker validates every instrumented source/header. It rejects malformed or
unknown directives, invalid iteration identifiers, unmatched or crossed pairs,
duplicate active names, and boundaries crossing C++ blocks or preprocessor
branches. Failures report the original file and line and prevent generation and
compilation. Generated files and source mappings remain unchanged on failure.
This checks profiler comments; the compiler still checks C++ expressions and types.

The profiling build expands these comments into Caliper guards under
`build/profiler/profiler-generated/`, then compiles those copies. Return,
exception, break, and continue close live timers through RAII. Normal
`macos-openmp` builds compile the original comment-only source. The build updates
`.profiler/regions.json` for report navigation; `#line` directives keep generated
compiler diagnostics tied to the original source locations.

`ProfilerRun` supplies the installed plugin's generator path automatically. To
configure from a terminal, pass `-DSPCRAFT_PROFILER_TOOLS=<profiler.nvim-checkout>`
to `cmake --preset profiler`, along with the dependency environment above. The
checkout supplies `cmake/Profiler.cmake`, `scripts/profiler_grammar.py`, and
`scripts/instrument.py`; Python 3.9+
is required during the profiling build. Unmatched markers fail the build before
compilation and report their original file and line.

**Space p → Build and run profiler** opens a profile picker. Select an existing
name/server pair to reuse it without typing, or choose **Create new profile…**.
The picker reads saved run manifests in `.profiler/runs/` and groups repeated runs
under the same name and server. Your last choice is remembered in
`.profiler/run-labels.json` and appears first, including after restarting Neovim.
Every measurement still gets a fresh timestamped output directory. Use **Build
and run with new profile name** (or `:ProfilerRun!`) to create another name.

Build and application output appear as `Profiler: Build` and `Profiler: Run`
Overseer tasks. On success, the profiler opens the timing report; four rank
traces, command logs, run labels, and `profile.json` are retained under
`.profiler/runs/`. Use Space p to load the latest report or reopen build/run
output. In the report, `r` filters ranks and `i` filters SUMMA stages; `q` closes
the report. Start another run from the profiler menu to get a fresh trace folder.

To check the same input through OMO's experiment runner, open **Space cE**,
press `O`, and choose **Local SUMMA smoke test**. Press `C` to configure, `B` to
build, or `r` to build and run. This experiment is self-contained and uses only
the local server. Your existing remote experiment is kept separately. Use the
profiler menu when collecting Caliper reports; experiment runs verify execution
without configuring trace collection.

## Define another serial action

Edit `.vscode/tasks.json`. To run a different executable, change the target in
both platform versions of Build's `args`, and change both Run executable paths.
`test_csr_matrix` is the initial smoke-test target and requires no input files.

To combine existing tasks, add an entry such as:

```json
{
  "label": "Configure, build and test",
  "dependsOn": ["Configure", "Build all", "Test"],
  "dependsOrder": "sequence"
}
```

The steps appear separately in Overseer's task panel. To pass arguments to a
program, add an `args` array to its process task. Use `options.cwd` and
`${workspaceFolder}` to keep paths portable. New named actions appear in
`:OverseerRun`; the fixed shortcuts use the task labels listed above.
