local M = {}
local options = { width = 66, auto_open = false }
local views = {}
local namespace = vim.api.nvim_create_namespace("solver_progress")
local guide_relative = "/docs/solver/implementation-order.md"

local function find_root(start)
  local directory = vim.fn.resolve(vim.fn.fnamemodify(start, ":p"))
  if vim.fn.isdirectory(directory) == 0 then
    directory = vim.fn.fnamemodify(directory, ":h")
  end
  while directory ~= "" do
    directory = directory:gsub("/$", "")
    if vim.fn.filereadable(directory .. guide_relative) == 1 then
      return directory
    end
    local parent = vim.fn.fnamemodify(directory, ":h")
    if parent == directory then
      break
    end
    directory = parent
  end
end

local function current_root()
  return find_root(vim.api.nvim_buf_get_name(0)) or find_root(vim.fn.getcwd())
end

local function render(view)
  if not vim.api.nvim_buf_is_valid(view.buf) then
    return
  end
  local ok, project = pcall(require("solver_progress.scan").project, view.root)
  if not ok then
    vim.notify(tostring(project), vim.log.levels.ERROR)
    return
  end
  view.project = project
  local total = #project.tasks
  local filled = math.floor(20 * project.done / total)
  local lines = {
    " Solver implementation progress",
    (" [%s%s] %d/%d implemented"):format(string.rep("#", filled), string.rep(".", 20 - filled), project.done, total),
    (" %d in progress | %d TODO | %d missing"):format(project.work, project.todo, project.missing),
    "",
    " Enter: source  d: guide  r: refresh  q: close",
    " Implemented = stub removed; numerical tests unverified.",
    "",
  }
  local labels = { scalar = "Week 1: scalar LU", supernodal = "Week 2: supernodal LU", mt = "Week 3: parallel LU" }
  local highlights = { done = "DiagnosticOk", work = "DiagnosticWarn", todo = "Comment", missing = "DiagnosticError" }
  local row_highlights = {}
  local previous_stage
  view.rows = {}
  local selected = vim.api.nvim_win_is_valid(view.win) and vim.api.nvim_win_get_cursor(view.win)[1] or 1
  local selected_id = view.selected_id
  local next_row
  for _, task in ipairs(project.tasks) do
    if task.stage ~= previous_stage then
      lines[#lines + 1] = " " .. labels[task.stage]
      row_highlights[#row_highlights + 1] = { #lines, "Title" }
      previous_stage = task.stage
    end
    local pointer = project.next_task == task and ">" or " "
    lines[#lines + 1] = ("%s %2d [%-7s] %s %s"):format(pointer, task.order, task.status, task.id, task.name)
    view.rows[#lines] = task
    row_highlights[#row_highlights + 1] = { #lines, highlights[task.status] }
    if project.next_task == task then
      next_row = #lines
    end
    if task.id == selected_id then
      selected = #lines
    end
    if task.reason then
      lines[#lines + 1] = "                  " .. task.reason
    end
  end
  lines[#lines + 1] = ""
  lines[#lines + 1] = project.next_task and (" Next: " .. project.next_task.name) or " All implementation stubs replaced."
  vim.bo[view.buf].modifiable = true
  vim.api.nvim_buf_set_lines(view.buf, 0, -1, false, lines)
  vim.api.nvim_buf_clear_namespace(view.buf, namespace, 0, -1)
  for _, row in ipairs(row_highlights) do
    vim.api.nvim_buf_set_extmark(view.buf, namespace, row[1] - 1, 0, { line_hl_group = row[2] })
  end
  vim.bo[view.buf].modifiable = false
  if vim.api.nvim_win_is_valid(view.win) then
    vim.api.nvim_win_set_cursor(view.win, { math.min(selected_id and selected or (next_row or selected), #lines), 0 })
  end
end

local function jump(view, guide)
  local row = vim.api.nvim_win_get_cursor(0)[1]
  local task = view.rows[row]
  if not task then
    return
  end
  if not guide and vim.fn.filereadable(task.file) == 0 then
    vim.notify("Source file not found: " .. task.file, vim.log.levels.WARN)
    return
  end
  view.selected_id = task.id
  if not vim.api.nvim_win_is_valid(view.source_win) or view.source_win == view.win then
    vim.cmd("leftabove vsplit")
    view.source_win = vim.api.nvim_get_current_win()
  else
    vim.api.nvim_set_current_win(view.source_win)
  end
  vim.cmd("edit " .. vim.fn.fnameescape(guide and view.project.guide or task.file))
  vim.api.nvim_win_set_cursor(0, { guide and task.guide_line or task.line, 0 })
  vim.cmd("normal! zz")
end

local function close(view)
  local focused = vim.api.nvim_get_current_win() == view.win
  if #vim.api.nvim_tabpage_list_wins(0) == 1 then
    vim.api.nvim_win_call(view.win, function() vim.cmd("enew") end)
  else
    vim.api.nvim_win_close(view.win, false)
  end
  if focused and vim.api.nvim_win_is_valid(view.source_win) then
    vim.api.nvim_set_current_win(view.source_win)
  end
end

function M.open(root)
  if root then
    root = find_root(root)
  else
    root = current_root()
  end
  if not root then
    vim.notify("Open SPCraft or pass its directory to :SolverProgress", vim.log.levels.WARN)
    return
  end
  local view = views[root]
  if view and vim.api.nvim_buf_is_valid(view.buf) and vim.api.nvim_win_is_valid(view.win) then
    vim.api.nvim_set_current_win(view.win)
    render(view)
    return
  end
  local source_win = vim.api.nvim_get_current_win()
  vim.cmd("botright vsplit")
  local win = vim.api.nvim_get_current_win()
  vim.api.nvim_win_set_width(win, options.width)
  local buf = vim.api.nvim_create_buf(false, true)
  vim.api.nvim_win_set_buf(win, buf)
  vim.api.nvim_buf_set_name(buf, "solver-progress://" .. root)
  vim.bo[buf].buftype = "nofile"
  vim.bo[buf].bufhidden = "wipe"
  vim.bo[buf].swapfile = false
  vim.bo[buf].filetype = "solver_progress"
  vim.wo[win].number = false
  vim.wo[win].relativenumber = false
  vim.wo[win].signcolumn = "no"
  vim.wo[win].wrap = false
  vim.wo[win].cursorline = true
  vim.wo[win].winfixwidth = true
  view = { root = root, buf = buf, win = win, source_win = source_win, rows = {} }
  views[root] = view
  local mapping_options = { buffer = buf, silent = true, nowait = true }
  vim.keymap.set("n", "<CR>", function() jump(view, false) end, mapping_options)
  vim.keymap.set("n", "d", function() jump(view, true) end, mapping_options)
  vim.keymap.set("n", "r", function()
    local selected = view.rows[vim.api.nvim_win_get_cursor(0)[1]]
    view.selected_id = selected and selected.id
    render(view)
  end, mapping_options)
  vim.keymap.set("n", "q", function() close(view) end, mapping_options)
  render(view)
end

function M.toggle(root)
  local target = root and find_root(root)
  if root and not target then
    M.open(root)
    return
  end
  for _, view in pairs(views) do
    if vim.api.nvim_buf_is_valid(view.buf) and vim.api.nvim_win_is_valid(view.win)
        and vim.api.nvim_win_get_tabpage(view.win) == vim.api.nvim_get_current_tabpage()
        and (not target or target == view.root) then
      close(view)
      return
    end
  end
  M.open(target)
end

function M.refresh()
  for _, view in pairs(views) do
    render(view)
  end
end

function M.setup(opts)
  options = vim.tbl_extend("force", options, opts or {})
  vim.api.nvim_create_user_command("SolverProgress", function(command)
    M.toggle(command.args ~= "" and command.args or nil)
  end, { nargs = "?", complete = "dir", desc = "Toggle SPCraft solver assignment progress", force = true })
  vim.api.nvim_create_user_command("SolverProgressRefresh", M.refresh,
    { desc = "Refresh solver assignment progress", force = true })
  local group = vim.api.nvim_create_augroup("SolverProgress", { clear = true })
  vim.api.nvim_create_autocmd("BufWritePost", {
    group = group,
    pattern = { "*/src/solver/*/Lu.cpp", "*/docs/solver/implementation-order.md" },
    callback = M.refresh,
  })
  vim.api.nvim_create_autocmd("BufEnter", {
    group = group,
    callback = function(event)
      if vim.bo[event.buf].filetype == "solver_progress" then
        M.refresh()
      end
    end,
  })
  local function auto_open()
    if options.auto_open and current_root() then
      vim.schedule(function() M.open() end)
    end
  end
  if vim.v.vim_did_enter == 1 then
    auto_open()
  else
    vim.api.nvim_create_autocmd("VimEnter", { group = group, once = true, callback = auto_open })
  end
end

return M
