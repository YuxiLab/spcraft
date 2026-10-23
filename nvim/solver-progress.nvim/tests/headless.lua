local function run()
  vim.o.swapfile = false
  vim.o.shadafile = "NONE"
  local script = debug.getinfo(1, "S").source:sub(2)
  local plugin_root = vim.fn.fnamemodify(script, ":p:h:h")
  local repository = vim.fn.fnamemodify(plugin_root, ":h:h")
  vim.opt.runtimepath:append(plugin_root)
  vim.cmd("runtime plugin/solver-progress.lua")
  local scanner = require("solver_progress.scan")
  local function status(body)
    return scanner.function_status("void Solve() {\n" .. body .. "\n}", "Solve", "W1.5")
  end
  local stub = 'throw std::logic_error("W1.5 Solve: implement permutations and triangular solves");'
  assert(status("// TODO(W1.5), Part 1: guide\n" .. stub) == "todo", "untouched stub")
  assert(status("auto count = 1;\n" .. stub) == "work", "partial implementation")
  assert(status("// TODO(W1.5): guide remains\nreturn;") == "done", "comments must not determine completion")
  assert(status("/* unfinished */") == "todo", "empty body")
  assert(status('auto s = "} /* not a comment */"; if (true) { return; }') == "done", "strings and nested braces")
  assert(status("auto c = '}'; auto count = 1'000;") == "done", "characters and digit separators")
  assert(status("auto c = L'}'; auto d = u8'{';") == "done", "prefixed characters")
  assert(status('auto text = R"tag(} " // arbitrary {)tag";\n' .. stub) == "work", "raw strings")
  assert(status('auto text = R"tag(' .. stub .. ')tag";') == "done", "stub text in a raw string is not a throw")
  assert(scanner.function_status("void Solve() {", "Solve", "W1.5") == "work", "unclosed body")
  assert(scanner.function_status("void Solve();", "Solve", "W1.5") == "missing", "declaration is not definition")
  assert(scanner.function_status("void SolveExtra() {}", "Solve", "W1.5") == "missing", "exact function name")

  local root = vim.fn.resolve(vim.fn.tempname())
  local function copy(relative)
    vim.fn.mkdir(vim.fn.fnamemodify(root .. "/" .. relative, ":h"), "p")
    vim.fn.writefile(vim.fn.readfile(repository .. "/" .. relative), root .. "/" .. relative)
  end
  copy("docs/solver/implementation-order.md")
  -- Own our numerical placeholders so these tests still work after students implement LU.
  local stages = {
    scalar = { "FactorColumn", "Factorize", "ForwardSolve", "BackwardSolve", "Solve" },
    supernodal = { "AssemblePanel", "UpdatePanel", "FactorPanel", "Factorize", "Solve" },
    mt = { "BuildTaskGraph", "Factorize" },
  }
  local weeks = { scalar = 1, supernodal = 2, mt = 3 }
  for stage, functions in pairs(stages) do
    local directory = root .. "/src/solver/" .. stage
    vim.fn.mkdir(directory, "p")
    local lines = {}
    for index, name in ipairs(functions) do
      lines[#lines + 1] = "void " .. name .. "() {"
      lines[#lines + 1] = ('  throw std::logic_error("W%d.%d %s: implement");'):format(weeks[stage], index, name)
      lines[#lines + 1] = "}"
    end
    vim.fn.writefile(lines, directory .. "/Lu.cpp")
  end
  local project = scanner.project(root)
  assert(#project.tasks == 12 and project.todo == 12, "all twelve scaffold tasks")
  assert(project.tasks[1].id == "W1.3" and project.tasks[3].id == "W1.1", "guide order, not source order")
  assert(project.tasks[12].id == "W3.2", "last task")

  vim.cmd("cd " .. vim.fn.fnameescape(root .. "/src/solver/scalar"))
  vim.cmd("edit Lu.cpp")
  local source_buf = vim.api.nvim_get_current_buf()
  vim.cmd("SolverProgress")
  local dashboard = vim.api.nvim_get_current_buf()
  local dashboard_win = vim.api.nvim_get_current_win()
  assert(vim.bo[dashboard].filetype == "solver_progress", "sidebar opened from nested directory")
  assert(vim.bo[dashboard].modifiable == false, "read-only sidebar")
  local function row_with(fragment)
    for line, text in ipairs(vim.api.nvim_buf_get_lines(dashboard, 0, -1, false)) do
      if text:find(fragment, 1, true) then return line, text end
    end
    error("Dashboard text missing: " .. fragment)
  end
  local selected_row = row_with("W1.3 scalar::ForwardSolve")
  assert(vim.api.nvim_win_get_cursor(0)[1] == selected_row, "next task initially selected")

  local function mapping(key)
    for _, entry in ipairs(vim.api.nvim_buf_get_keymap(dashboard, "n")) do
      if entry.lhs == key then return entry.callback end
    end
    error("Mapping missing: " .. key)
  end
  mapping("<CR>")()
  assert(vim.api.nvim_get_current_buf() == source_buf, "Enter opens source")
  local current_line = vim.api.nvim_get_current_line()
  assert(current_line:find("void ForwardSolve", 1, true), "Enter selects function definition")
  local source_lines = vim.api.nvim_buf_get_lines(source_buf, 0, -1, false)
  for line, text in ipairs(source_lines) do
    if text:find('throw std::logic_error("W1.3 ', 1, true) then
      source_lines[line] = "  return; // Test fixture only, not a numerical implementation."
    end
  end
  vim.api.nvim_buf_set_lines(source_buf, 0, -1, false, source_lines)
  assert(scanner.project(root).done == 1, "loaded unsaved edits are scanned")
  vim.cmd("write")
  assert(select(2, row_with("W1.3 scalar::ForwardSolve")):find("[done", 1, true), "save refreshes sidebar")
  row_with("1/12 implemented")

  vim.api.nvim_set_current_win(dashboard_win)
  selected_row = row_with("W1.3 scalar::ForwardSolve")
  vim.api.nvim_win_set_cursor(0, { selected_row, 0 })
  mapping("d")()
  assert(vim.api.nvim_buf_get_name(0) == root .. "/docs/solver/implementation-order.md", "guide navigation")
  assert(vim.api.nvim_get_current_line():find("scalar::ForwardSolve", 1, true), "guide navigation selects matching row")

  local mt = root .. "/src/solver/mt/Lu.cpp"
  vim.fn.delete(mt)
  assert(scanner.project(root).missing == 2, "missing file marks both functions")
  vim.api.nvim_set_current_win(dashboard_win)
  mapping("q")()
  vim.cmd("SolverProgress " .. vim.fn.fnameescape(root))
  assert(vim.bo.filetype == "solver_progress", "closed sidebar can reopen")
  require("solver_progress").open()
  local count = 0
  for _, buf in ipairs(vim.api.nvim_list_bufs()) do
    if vim.api.nvim_buf_is_valid(buf) and vim.bo[buf].filetype == "solver_progress" then count = count + 1 end
  end
  assert(count == 1, "opening twice reuses sidebar")
  local toggle_win = vim.api.nvim_get_current_win()
  vim.cmd("SolverProgress")
  assert(not vim.api.nvim_win_is_valid(toggle_win), "toggle closes focused sidebar")
  assert(vim.bo.filetype ~= "solver_progress", "toggle returns to source window")
  local editing_win = vim.api.nvim_get_current_win()
  vim.api.nvim_win_set_buf(editing_win, source_buf)
  vim.api.nvim_buf_set_lines(source_buf, 0, 0, false, { "// Unsaved test edit" })
  vim.cmd("SolverProgress")
  toggle_win = vim.api.nvim_get_current_win()
  assert(vim.bo.filetype == "solver_progress", "toggle reopens sidebar")
  vim.api.nvim_set_current_win(editing_win)
  vim.cmd("SolverProgress")
  assert(not vim.api.nvim_win_is_valid(toggle_win), "toggle closes sidebar from source window")
  assert(vim.api.nvim_get_current_win() == editing_win, "toggle keeps source focus")
  assert(vim.bo[source_buf].modified, "toggle preserves unsaved source edits")
  assert(vim.api.nvim_buf_get_lines(source_buf, 0, 1, false)[1] == "// Unsaved test edit", "source data retained")
  vim.cmd("SolverProgress")
  vim.cmd("only")
  vim.cmd("SolverProgress")
  assert(vim.bo.filetype ~= "solver_progress", "last-window toggle leaves an editor window")
  vim.fn.delete(root, "rf")
  print("solver-progress: scanner, ordering, navigation, refresh, and toggle checks passed")
end

local ok, reason = xpcall(run, debug.traceback)
if not ok then
  vim.api.nvim_err_writeln(reason)
  vim.cmd("cquit 1")
else
  vim.cmd("qa!")
end
