# Solver progress for Neovim

An SPCraft assignment sidebar with the recommended implementation order, progress
counts, the next unfinished function, and source/guide navigation. It reads the
tables in `docs/solver/implementation-order.md`, so the guide owns the order.
No third-party Neovim plugins, Tree-sitter parsers, or build tools are required.
Use Neovim 0.9 or newer.

```text
Solver implementation progress
[#####...............] 3/12 implemented
1 in progress | 8 TODO | 0 missing

Week 1: scalar LU
   1 [done   ] W1.3 scalar::ForwardSolve
   2 [done   ] W1.4 scalar::BackwardSolve
   3 [done   ] W1.1 scalar::FactorColumn
>  4 [work   ] W1.2 scalar::Factorize
   5 [todo   ] W1.5 scalar::Solve
...
```

## Try it from this checkout

From the SPCraft repository root:

```sh
nvim --cmd 'set runtimepath+=./nvim/solver-progress.nvim' -c SolverProgress
```

This opens the sidebar immediately. You can also open it later with
`:SolverProgress` from any file or subdirectory in this repository. Pass an
explicit project directory when working elsewhere: `:SolverProgress /path/to/spcraft`.

## Install with lazy.nvim / NvChad

Add this entry to your plugin specifications and replace the directory with
your checkout's path on the current machine:

```lua
{
  dir = vim.fn.expand("~/path/to/spcraft/nvim/solver-progress.nvim"),
  name = "solver-progress.nvim",
  lazy = false,
  config = function()
    require("solver_progress").setup({
      width = 66,
      auto_open = false, -- Set true to open when starting Neovim in SPCraft.
    })
  end,
  keys = {
    { "<leader>sp", "<cmd>SolverProgress<cr>", desc = "Solver progress" },
  },
}
```

For a plain `init.lua`, append the plugin directory to `vim.opt.runtimepath`
and call `require("solver_progress").setup()` with the same options. Neovim's
normal plugin loading registers the commands. No machine-specific paths are
stored in the plugin itself.

## Use the sidebar

| Key / command | Action |
|---|---|
| `:SolverProgress` | Toggle the sidebar; select the next unfinished task when opening. |
| Space, `s`, `p` | Toggle the sidebar with the NvChad/lazy.nvim mapping above (Space leader). |
| Enter | Open the selected function in the source window beside the sidebar. |
| `d` | Open the selected row in the implementation-order guide. |
| `r` | Refresh, including unsaved changes in loaded source buffers. |
| `q` | Close the sidebar. |
| `:SolverProgressRefresh` | Refresh an open sidebar from another window. |

The sidebar refreshes automatically after saving a solver `Lu.cpp` or the
implementation-order guide, and when you return to the sidebar. It never builds
or runs the solver, and does not write progress state into your source files.
The toggle works from the sidebar or the source window beside it, preserving
unsaved source edits. Calling `require("solver_progress").open()` from Lua
still explicitly opens or focuses the sidebar without toggling it closed.

## How status is detected

| Status | Meaning |
|---|---|
| `todo` | The function contains only its original `throw std::logic_error("W…")` placeholder, or has an empty body. |
| `work` | Other code exists alongside the placeholder, or the body is currently unclosed. |
| `done` | The body contains code and its assignment placeholder has been removed. |
| `missing` | The source file or expected function definition cannot be found. |

Comments, nested braces, normal strings, character literals, and C++ raw strings
are skipped when locating function boundaries. The teaching signatures have no
trailing qualifiers after the parameter list. Keep the assignment ID in the
placeholder while work is unfinished; remove the placeholder when you finish.
The detailed `TODO(W…)` instructional comments can stay in place.

**`done` means implementation present, not numerical correctness or passing tests.**
Run the week's acceptance tests and additional kernel checks before submitting.
The plugin does not evaluate conditional compilation or arbitrary C++ semantics.

## Verify the plugin

From the SPCraft repository root:

```sh
nvim --headless -u NONE -c 'lua dofile("nvim/solver-progress.nvim/tests/headless.lua")'
```

The checks exercise status detection, document ordering, missing definitions,
unsaved edits, automatic refresh after saving, sidebar navigation, and toggling.
They operate on a temporary project without modifying the assignment sources.
