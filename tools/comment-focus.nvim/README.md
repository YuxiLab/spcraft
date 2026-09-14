# comment-focus.nvim

Read the explanation without the code competing for your attention. Toggle
comment focus to dim code and render comments in a bright, bold foreground.
Function and class signatures stay visible too, so you can follow the
structure while reading comments.
Toggle again to restore normal highlighting. Text and colorscheme definitions
are preserved; only the plugin's own highlights are added and removed.

Requires Neovim 0.11 or newer. Uses Tree-sitter `@comment` captures (including
documentation comments and injected languages), with Vim syntax highlighting
as a fallback when a parser or highlight query is unavailable. Enable
`syntax on` for that fallback. A language needs one of these highlighting
sources; plain text has no comment syntax to recognize.

## Install locally

This directory is a standalone plugin. Copy it into your Neovim package
directory, running these commands from this directory:

```sh
data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
mkdir -p "$data_home/nvim/site/pack/local/start/comment-focus.nvim"
cp -R lua plugin queries README.md "$data_home/nvim/site/pack/local/start/comment-focus.nvim/"
```

Restart Neovim. No plugin manager or setup call is required.

To try it without installing, run from this directory:

```sh
nvim --cmd 'set runtimepath+=.' -c 'runtime plugin/comment-focus.lua' README.md
```

Then open a source file with `:edit`. For lazy.nvim, use a local plugin spec
with `dir` pointing to this directory and `opts = {}`.

## Hotkey

**`<leader>tc`** toggles focus in the current buffer. With a space leader,
press **Space, t, c** in normal mode. With Neovim's default leader, press
**backslash, t, c**. Set `vim.g.mapleader` before loading plugins. An existing
mapping is preserved; use the command or configure a different key in that case.

Commands: `:CommentFocusToggle`, `:CommentFocusEnable`, `:CommentFocusDisable`.
All windows showing the same buffer share its focus state. Other buffers are
unaffected. Focus starts disabled and does not modify your files.

## Configure

Call this from your configuration (or use these options in lazy.nvim):

```lua
require('comment-focus').setup({
  keymap = '<leader>tc', -- false disables the default key
  dim = 0.25,           -- 0 = background color; 1 = normal foreground
  bold = true,
  show_signatures = true, -- keep full function and class signatures visible
  -- comment_color = '#ffe08a', -- defaults to Normal's foreground
  -- signature_color = '#98c379', -- defaults to the comment color
  debounce_ms = 100,
  priority = 1000,      -- above standard syntax and semantic highlights
})
```

For your own mapping, use `<Plug>(CommentFocusToggle)` or
`require('comment-focus').toggle()`. `is_enabled()` returns the current state.

Colors follow the current colorscheme, including light backgrounds. Enable
`termguicolors` for smooth dimming; terminal-color mode uses gray and a bright
comment color. With a transparent background, the dim blend assumes black for
dark themes and white for light themes.

The plugin refreshes comment ranges after edits, with a short debounce. The
syntax fallback scans the buffer by byte and can be slower on large files;
Tree-sitter is preferred. Search matches and selection can still overlay the
focus highlights. This is a reading aid, not concealment.

Signatures use separate Tree-sitter queries for C, C++, Lua, and Python.
They include return types, parameters, qualifiers, template headers, class
inheritance, and Python decorators, including signatures spanning several lines.
C++ includes methods, constructors, and destructors. C/C++ struct, union, and enum
headers are also kept visible. Highlighting stops at the opening brace or before
the first body statement. Function calls and bodies stay dim. Other languages and
the Vim syntax fallback retain comment-only focus. Add a
`queries/<language>/comment_focus.scm` query with `@function` and `@class`
captures on declaration names to identify their enclosing signatures. Set
`show_signatures = false` for comments only.

The implementation uses Neovim's documented
[Tree-sitter queries](https://neovim.io/doc/user/treesitter/) and
[extmark highlights](https://neovim.io/doc/user/api/#nvim_buf_set_extmark()).

## Test

From this directory, with C, C++, Lua, and Python parsers installed:

```sh
nvim --headless -u NONE -l tests/run.lua
```
