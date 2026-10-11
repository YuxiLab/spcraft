# Neovim build and run workflows

The repository provides `.vscode/tasks.json`, which Overseer discovers without
loading project Lua code. It demonstrates sequential CMake actions and error
navigation. Install `stevearc/overseer.nvim` in your Neovim plugin manager and run
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
| Space cp | Pick any task, including Run and Test |
| Space cO / Space c0 | Toggle the task/output panel (capital O or zero) |
| Space co / Space cq | Open / close quickfix |
| `]q` / `[q` | Next / previous error |

Build shortcuts save modified files before starting. In the task panel, `o`
opens the selected task's output, and Enter opens actions such as restart or stop.
Failed process tasks open quickfix and their full output. Compiler messages with
source locations become navigable errors and editor diagnostics. A nonzero
program exit stops a workflow even if its output has no source location.

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
