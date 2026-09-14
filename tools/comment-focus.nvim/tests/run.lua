vim.opt.runtimepath:prepend('.')
vim.g.mapleader = ' '
vim.cmd('filetype on')
vim.cmd('syntax on')
vim.cmd('runtime plugin/comment-focus.lua')
local api = vim.api
local focus = require('comment-focus')
local ns = api.nvim_get_namespaces()['comment-focus']
local checks = 0

local function check(condition, message)
  assert(condition, message)
  checks = checks + 1
end

local function buffer(filetype, lines)
  local buf = api.nvim_create_buf(true, false)
  api.nvim_set_current_buf(buf)
  vim.bo.filetype = filetype
  api.nvim_buf_set_lines(buf, 0, -1, false, lines)
  return buf
end

local function marks(buf)
  return api.nvim_buf_get_extmarks(buf, ns, 0, -1, { details = true })
end

local function has_highlight(buf, row, col, group)
  for _, mark in ipairs(marks(buf)) do
    local sr, sc, details = mark[2], mark[3], mark[4]
    local er, ec = details.end_row, details.end_col
    if details.hl_group == group
      and (row > sr or (row == sr and col >= sc))
      and (row < er or (row == er and col < ec)) then
      return true
    end
  end
  return false
end

local function is_comment(buf, row, col)
  return has_highlight(buf, row, col, 'CommentFocusComment')
end

check(vim.fn.maparg(' tc', 'n') == '<Plug>(CommentFocusToggle)', 'default hotkey missing')
local source = {
  'const char *s = "// not a comment"; // actual comment',
  '/* multiline',
  '   explanation */ int x = 1;',
  '// UTF-8: café 中文',
}
local buf = buffer('c', source)
local normal = api.nvim_get_hl(0, { name = 'Normal', link = false })
check(not focus.is_enabled(), 'must start disabled')
vim.cmd('CommentFocusEnable')
check(focus.is_enabled(), 'enable command failed')
check(not is_comment(buf, 0, 18), 'comment marker inside string was highlighted')
check(is_comment(buf, 0, 39), 'inline comment missing')
check(is_comment(buf, 1, 3) and is_comment(buf, 2, 5), 'multiline comment missing')
check(not is_comment(buf, 2, 22), 'code after block comment highlighted')
check(is_comment(buf, 3, #source[4] - 1), 'UTF-8 comment truncated')
check(vim.deep_equal(normal, api.nvim_get_hl(0, { name = 'Normal', link = false })),
  'Normal highlight modified')
check(vim.deep_equal(source, api.nvim_buf_get_lines(buf, 0, -1, false)), 'buffer text modified')
local code_mark = marks(buf)[1][4]
check(code_mark.hl_group == 'CommentFocusCode' and code_mark.priority == 1000,
  'code dim overlay missing')
check(code_mark.end_row == #source, 'code overlay does not cover whole buffer')

-- Buffer-local state must survive switching buffers and opening splits.
local other = buffer('lua', { 'local n = 1 -- explanation' })
check(not focus.is_enabled() and #marks(other) == 0, 'focus leaked to another buffer')
focus.enable()
check(is_comment(other, 0, 15) and not is_comment(other, 0, 3), 'Lua comment detection failed')
focus.disable()
api.nvim_set_current_buf(buf)
vim.cmd('split')
check(focus.is_enabled(), 'split lost focus state')
vim.cmd('close')

-- A real edit must remove stale comment ranges and expand the dim range.
api.nvim_buf_set_lines(buf, 0, -1, false, { 'int value = 2;', '// replacement', 'int y;', '// last', 'int z;' })
api.nvim_exec_autocmds('TextChanged', { buffer = buf })
check(vim.wait(1000, function()
  return not is_comment(buf, 0, 5) and is_comment(buf, 1, 3) and is_comment(buf, 3, 3)
end), 'edit refresh failed')
check(marks(buf)[1][4].end_row == 5, 'code dim range did not expand after editing')
vim.cmd('colorscheme default')
check(api.nvim_get_hl(0, { name = 'CommentFocusComment' }).fg ~= nil,
  'colorscheme removed focus colors')
vim.cmd('CommentFocusDisable')
check(not focus.is_enabled() and #marks(buf) == 0, 'disable left decorations')

-- Exercise the actual normal-mode mapping, not just the Lua function.
api.nvim_feedkeys(' tc', 'xt', false)
check(focus.is_enabled(), 'hotkey did not enable focus')
api.nvim_feedkeys(' tc', 'xt', false)
check(not focus.is_enabled(), 'hotkey did not disable focus')

-- Syntax fallback has no parser; contained TODO remains part of the comment.
local fallback = buffer('comment_focus_test', {
  'let value = "# string" # note TODO café',
  '/* multi',
  '   line */ value',
})
vim.cmd('syntax clear')
vim.cmd([[syntax region TestString start=/"/ end=/"/]])
vim.cmd([[syntax match TestComment /#.*/ contains=TestTodo]])
vim.cmd([[syntax region TestBlockComment start=/\/\*/ end=/\*\//]])
vim.cmd([[syntax keyword TestTodo TODO contained]])
vim.cmd('highlight link TestComment Comment')
vim.cmd('syntax sync fromstart')
focus.enable()
check(not is_comment(fallback, 0, 14), 'syntax fallback mistook string for comment')
check(is_comment(fallback, 0, 24) and is_comment(fallback, 0, 30), 'fallback missed comment/TODO')
check(is_comment(fallback, 1, 3) and is_comment(fallback, 2, 3), 'fallback missed block comment')
check(not is_comment(fallback, 2, 13), 'fallback highlighted trailing code')

-- Pending callbacks must not revive focus after it is disabled or wiped.
api.nvim_exec_autocmds('TextChanged', { buffer = fallback })
focus.disable()
vim.wait(150, function() return false end)
check(#marks(fallback) == 0, 'pending callback revived disabled focus')
focus.enable()
api.nvim_exec_autocmds('TextChanged', { buffer = fallback })
api.nvim_buf_delete(fallback, { force = true })
vim.wait(150, function() return false end)
check(not focus.is_enabled(fallback), 'wiped buffer retained state')

local empty = buffer('', { '' })
focus.enable()
check(#marks(empty) == 1, 'empty buffer failed')
focus.disable()
focus.setup({ keymap = '<leader>cf', comment_color = '#ffe08a', dim = 0.4 })
check(vim.fn.maparg(' tc', 'n') == '', 'old mapping left behind')
check(vim.fn.maparg(' cf', 'n') ~= '', 'custom mapping missing')
check(api.nvim_get_hl(0, { name = 'CommentFocusComment' }).fg == 0xffe08a,
  'custom comment color ignored')
focus.setup({ keymap = false })
check(vim.fn.maparg(' cf', 'n') == '', 'keymap=false did not remove mapping')
vim.keymap.set('n', '<leader>tc', ':echo "existing"<CR>')
local existing = vim.fn.maparg(' tc', 'n')
focus.setup({})
check(vim.fn.maparg(' tc', 'n') == existing, 'existing mapping overwritten')

-- Full signatures are visible; calls and body statements stay dim.
focus.setup({ comment_color = '#98c379' })
local declarations = {
  'template<class T> class Matrix {',
  'public:',
  '  Matrix();',
  '  ~Matrix();',
  '  T multiply(T input) const;',
  '  T operator+(T other) const;',
  '};',
  'template<class T> T Matrix<T>::multiply(T input) const {',
  '  // Scale the input.',
  '  return helper(input);',
  '}',
  'struct Scratch { int value; };',
  'int *allocate(int count);',
  'int (*callback)(int);',
}
local cpp = buffer('cpp', declarations)
focus.enable()
local function signature_at(row, word)
  local col = assert(declarations[row]:find(word, 1, true)) - 1
  return has_highlight(cpp, row - 1, col, 'CommentFocusSignature')
end
for _, sample in ipairs({
  { 1, 'Matrix' }, { 3, 'Matrix' }, { 4, '~Matrix' }, { 5, 'multiply' },
  { 6, 'operator+' }, { 8, 'Matrix<T>::multiply' }, { 12, 'Scratch' }, { 13, 'allocate' },
  { 1, 'template' }, { 1, 'T' }, { 5, 'input' }, { 5, 'const' }, { 13, 'count' }, { 13, 'int' },
}) do
  check(signature_at(sample[1], sample[2]), 'C++ name missing: ' .. sample[2])
end
for _, sample in ipairs({
  { 10, 'helper' }, { 10, 'input' }, { 12, 'value' }, { 14, 'callback' },
}) do
  check(not signature_at(sample[1], sample[2]), 'Non-declaration brightened: ' .. sample[2])
end
check(is_comment(cpp, 8, 5), 'name highlighting displaced comments')
check(api.nvim_get_hl(0, { name = 'CommentFocusSignature' }).fg == 0x98c379,
  'names did not inherit green comment color')
focus.setup({ show_signatures = false })
check(not signature_at(1, 'Matrix') and is_comment(cpp, 8, 5), 'comments-only option failed')
focus.setup({ signature_color = '#abcdef' })
check(signature_at(1, 'Matrix') and api.nvim_get_hl(0, { name = 'CommentFocusSignature' }).fg == 0xabcdef,
  'custom name color failed')
api.nvim_buf_set_lines(cpp, 2, 3, false, { '  int field;' })
api.nvim_exec_autocmds('TextChanged', { buffer = cpp })
check(vim.wait(1000, function()
  return not has_highlight(cpp, 2, 4, 'CommentFocusSignature')
end), 'edit retained a stale declaration highlight')
focus.disable()
check(#marks(cpp) == 0, 'disable left name highlights')

local py = buffer('python', { 'class Solver:', '    def solve(self, rhs):', '        return solve(rhs)' })
focus.enable()
check(has_highlight(py, 0, 6, 'CommentFocusSignature'), 'Python class name missing')
check(has_highlight(py, 1, 8, 'CommentFocusSignature'), 'Python method name missing')
check(not has_highlight(py, 2, 15, 'CommentFocusSignature'), 'Python call brightened')
focus.disable()
local lua = buffer('lua', { 'function M.solve(rhs)', '  return solve(rhs)', 'end' })
focus.enable()
check(has_highlight(lua, 0, 11, 'CommentFocusSignature'), 'Lua function name missing')
check(not has_highlight(lua, 1, 9, 'CommentFocusSignature'), 'Lua call brightened')
focus.disable()
local c = buffer('c', {
  'struct Data { int count; };', 'int work(int n) { return work(n); }',
  'struct Data *data;', 'struct Forward;',
})
focus.enable()
check(has_highlight(c, 0, 7, 'CommentFocusSignature'), 'C struct name missing')
check(has_highlight(c, 1, 5, 'CommentFocusSignature'), 'C function name missing')
check(not has_highlight(c, 1, 26, 'CommentFocusSignature'), 'C call brightened')
check(not has_highlight(c, 2, 7, 'CommentFocusSignature'), 'C struct reference brightened')
check(has_highlight(c, 3, 7, 'CommentFocusSignature'), 'C forward declaration missing')
focus.disable()

local multiline = buffer('cpp', {
  'template<class T>',
  'class Matrix final',
  '  : public Base<T> {',
  '  int field;',
  '};',
  'template<class T>',
  '[[nodiscard]] auto multiply(',
  '    const Matrix<T>& lhs,',
  '    const Matrix<T>& rhs) noexcept -> T',
  '{ return compute(lhs, rhs); }',
})
focus.enable()
for _, sample in ipairs({ { 0, 0 }, { 1, 13 }, { 2, 4 }, { 5, 0 }, { 6, 1 }, { 7, 4 }, { 8, 25 }, { 9, 0 } }) do
  check(has_highlight(multiline, sample[1], sample[2], 'CommentFocusSignature'),
    'multiline signature missing at row ' .. sample[1])
end
check(not has_highlight(multiline, 3, 6, 'CommentFocusSignature'), 'class field brightened')
check(not has_highlight(multiline, 9, 2, 'CommentFocusSignature'), 'same-line function body brightened')
focus.disable()

local decorated = buffer('python', {
  '@decorate', 'class Solver(Base):', '    value = 1',
  '@decorate', 'def solve(', '    rhs: int,', ') -> int:', '    return rhs',
})
focus.enable()
for _, sample in ipairs({ { 0, 0 }, { 1, 13 }, { 3, 0 }, { 4, 0 }, { 5, 4 }, { 6, 5 } }) do
  check(has_highlight(decorated, sample[1], sample[2], 'CommentFocusSignature'),
    'Python signature missing at row ' .. sample[1])
end
check(not has_highlight(decorated, 2, 4, 'CommentFocusSignature'), 'Python class body brightened')
check(not has_highlight(decorated, 7, 4, 'CommentFocusSignature'), 'Python function body brightened')
focus.disable()

print(('comment-focus: %d checks passed'):format(checks))
vim.cmd('qa!')
