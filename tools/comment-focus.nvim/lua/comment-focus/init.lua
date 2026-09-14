local M = {}
local api = vim.api
local ns = api.nvim_create_namespace('comment-focus')
local active = {}
local defaults = {
  keymap = '<leader>tc',
  dim = 0.25, -- Fraction of Normal's foreground retained against its background.
  comment_color = nil,
  show_signatures = true,
  signature_color = nil, -- Defaults to the comment color.
  bold = true,
  debounce_ms = 100,
  priority = 1000,
}
local config = vim.deepcopy(defaults)
local installed_key
local initialized = false

local function colors()
  local normal = api.nvim_get_hl(0, { name = 'Normal', link = false })
  local dark = vim.o.background == 'dark'
  local fg = normal.fg or (dark and 0xeeeeee or 0x202020)
  local bg = normal.bg or (dark and 0x000000 or 0xffffff)
  local dim = 0
  for _, shift in ipairs({ 16, 8, 0 }) do
    local f = math.floor(fg / 2 ^ shift) % 256
    local b = math.floor(bg / 2 ^ shift) % 256
    dim = dim + math.floor(b + (f - b) * config.dim + 0.5) * 2 ^ shift
  end
  api.nvim_set_hl(0, 'CommentFocusCode', {
    fg = dim, ctermfg = dark and 8 or 7, bold = false, italic = false,
  })
  api.nvim_set_hl(0, 'CommentFocusComment', {
    fg = config.comment_color or fg, ctermfg = dark and 15 or 0,
    bold = config.bold, italic = false,
  })
  api.nvim_set_hl(0, 'CommentFocusSignature', {
    fg = config.signature_color or config.comment_color or fg,
    ctermfg = dark and 15 or 0, bold = config.bold, italic = false,
  })
end

local function focus_mark(buf, sr, sc, er, ec, group)
  api.nvim_buf_set_extmark(buf, ns, sr, sc, {
    end_row = er, end_col = ec,
    hl_group = group or 'CommentFocusComment', priority = config.priority + 1,
    strict = false,
  })
end

local function is_type_declaration(node)
  local specifier = node:parent()
  if not specifier or not specifier:type():match('_specifier$') then
    return true
  end
  if #specifier:field('body') > 0 then
    return true
  end
  -- C uses the same node for "struct Data;" and "struct Data *value;".
  local declaration = specifier:parent()
  if declaration then
    for _, type_node in ipairs(declaration:field('type')) do
      if type_node:id() == specifier:id() then
        return false
      end
    end
  end
  return true
end

local function signature_range(node, capture)
  local declaration = node:parent()
  if capture == 'function' then
    -- Walk through C/C++ pointer and function declarators to the return type.
    while declaration do
      local kind = declaration:type()
      if kind == 'function_definition' or kind == 'function_declaration'
        or kind == 'declaration' or kind == 'field_declaration' then
        break
      end
      declaration = declaration:parent()
    end
  end
  if not declaration then
    return node:range()
  end
  local sr, sc, er, ec = declaration:range()
  local body = declaration:field('body')[1]
  if body then
    local kind = body:type()
    if kind == 'compound_statement' or kind == 'field_declaration_list'
      or kind == 'enumerator_list' or kind == 'block' then
      er, ec = body:start()
      -- Keep an opening brace, but never brighten the first body statement.
      if kind ~= 'block' then
        ec = ec + 1
      end
    end
  end
  -- Template parameters and Python decorators belong to the visible header.
  local parent = declaration:parent()
  while parent and (parent:type() == 'template_declaration'
    or parent:type() == 'decorated_definition') do
    sr, sc = parent:start()
    parent = parent:parent()
  end
  return sr, sc, er, ec
end

local function treesitter_highlights(buf)
  local parser = vim.treesitter.get_parser(buf, nil, { error = false })
  if not parser then
    return false
  end
  -- Include injected languages (for example, C in a Markdown fence).
  parser:parse(true)
  local supported = false
  parser:for_each_tree(function(tree, language)
    -- Find declarations, then expand their names through the full signature.
    local names = config.show_signatures
      and vim.treesitter.query.get(language:lang(), 'comment_focus')
    if names then
      for id, node in names:iter_captures(tree:root(), buf, 0, -1) do
        local capture = names.captures[id]
        if capture == 'function' or (capture == 'class' and is_type_declaration(node)) then
          local sr, sc, er, ec = signature_range(node, capture)
          focus_mark(buf, sr, sc, er, ec, 'CommentFocusSignature')
        end
      end
    end
    local query = vim.treesitter.query.get(language:lang(), 'highlights')
    if not query then
      return
    end
    supported = true
    for id, node, metadata in query:iter_captures(tree:root(), buf, 0, -1) do
      local name = query.captures[id]
      if name == 'comment' or name:match('^comment%.') then
        -- Honor offsets in highlight queries; node:range() alone ignores them.
        local range = vim.treesitter.get_range(node, buf, metadata[id])
        focus_mark(buf, range[1], range[2], range[4], range[5])
      end
    end
  end)
  return supported
end

local function syntax_comments(buf)
  -- synstack is window-contextual. buf_call selects a window for this buffer.
  api.nvim_buf_call(buf, function()
    local lines = api.nvim_buf_get_lines(buf, 0, -1, false)
    for row, line in ipairs(lines) do
      local start
      for col = 1, #line + 1 do
        local comment = false
        if col <= #line then
          for _, id in ipairs(vim.fn.synstack(row, col)) do
            local name = vim.fn.synIDattr(id, 'name'):lower()
            local linked = vim.fn.synIDattr(vim.fn.synIDtrans(id), 'name'):lower()
            if name:find('comment', 1, true) or linked == 'comment' then
              comment = true
              break
            end
          end
        end
        if comment and not start then
          start = col - 1
        elseif not comment and start then
          focus_mark(buf, row - 1, start, row - 1, col - 1)
          start = nil
        end
      end
    end
  end)
end

local function refresh(buf)
  if not active[buf] or not api.nvim_buf_is_loaded(buf) then
    return
  end
  api.nvim_buf_clear_namespace(buf, ns, 0, -1)
  api.nvim_buf_set_extmark(buf, ns, 0, 0, {
    end_row = api.nvim_buf_line_count(buf), end_col = 0,
    hl_group = 'CommentFocusCode', hl_eol = true, priority = config.priority,
  })
  if not treesitter_highlights(buf) then
    syntax_comments(buf)
  end
end

local function queue_refresh(buf)
  local state = active[buf]
  if not state then
    return
  end
  state.generation = state.generation + 1
  local generation = state.generation
  vim.defer_fn(function()
    -- Disabling, re-enabling, or wiping a buffer invalidates pending work.
    if active[buf] == state and state.generation == generation then
      refresh(buf)
    end
  end, config.debounce_ms)
end

function M.is_enabled(buf)
  return active[buf or api.nvim_get_current_buf()] ~= nil
end

function M.enable()
  if not initialized then
    M.setup()
  end
  local buf = api.nvim_get_current_buf()
  if vim.bo[buf].buftype ~= '' then
    return
  end
  active[buf] = { generation = 0 }
  refresh(buf)
end

function M.disable()
  local buf = api.nvim_get_current_buf()
  active[buf] = nil
  api.nvim_buf_clear_namespace(buf, ns, 0, -1)
end

function M.toggle()
  if M.is_enabled() then
    M.disable()
  else
    M.enable()
  end
end

function M.setup(opts)
  config = vim.tbl_deep_extend('force', vim.deepcopy(defaults), opts or {})
  assert(type(config.dim) == 'number' and config.dim >= 0 and config.dim <= 1,
    'comment-focus: dim must be between 0 and 1')
  assert(type(config.debounce_ms) == 'number' and config.debounce_ms >= 0,
    'comment-focus: debounce_ms must be non-negative')
  colors()
  local group = api.nvim_create_augroup('CommentFocus', { clear = true })
  api.nvim_create_autocmd({ 'TextChanged', 'TextChangedI', 'TextChangedP', 'BufEnter', 'FileType', 'Syntax' }, {
    group = group,
    callback = function(event) queue_refresh(event.buf) end,
  })
  api.nvim_create_autocmd({ 'BufUnload', 'BufWipeout' }, {
    group = group,
    callback = function(event) active[event.buf] = nil end,
  })
  api.nvim_create_autocmd('ColorScheme', { group = group, callback = colors })
  api.nvim_create_autocmd('OptionSet', {
    group = group, pattern = 'background', callback = colors,
  })
  for suffix, action in pairs({ Toggle = M.toggle, Enable = M.enable, Disable = M.disable }) do
    api.nvim_create_user_command('CommentFocus' .. suffix, action, {})
  end
  vim.keymap.set('n', '<Plug>(CommentFocusToggle)', M.toggle,
    { desc = 'Toggle comment focus' })
  if installed_key then
    local previous = vim.fn.maparg(installed_key, 'n', false, true)
    if previous.rhs == '<Plug>(CommentFocusToggle)' then
      vim.keymap.del('n', installed_key)
    end
    installed_key = nil
  end
  if config.keymap and config.keymap ~= '' and vim.fn.maparg(config.keymap, 'n') == '' then
    -- Remember the expanded leader so a later setup can remove the right map.
    installed_key = config.keymap:gsub('<leader>', function() return vim.g.mapleader or '\\' end)
      :gsub('<localleader>', function() return vim.g.maplocalleader or '\\' end)
    vim.keymap.set('n', installed_key, '<Plug>(CommentFocusToggle)',
      { desc = 'Toggle comment focus' })
  end
  initialized = true
  for buf in pairs(active) do
    refresh(buf)
  end
end

function M._load()
  if not initialized then
    M.setup()
  end
end

return M
