local M = {}

-- Preserve byte positions while hiding comments and string contents from brace matching.
local function mask_source(text)
  local code, structure = {}, {}
  local i = 1
  while i <= #text do
    local char = text:sub(i, i)
    local pair = text:sub(i, i + 1)
    local finish
    local comment = false
    if pair == "//" then
      finish = (text:find("\n", i + 2, true) or (#text + 1)) - 1
      comment = true
    elseif pair == "/*" then
      local closing = text:find("*/", i + 2, true)
      finish = closing and closing + 1 or #text
      comment = true
    elseif char == '"' or char == "'" then
      -- C++ digit separators are not character literals.
      local before = text:sub(1, i - 1)
      local character_prefix = before:match("%f[%w_][LuU]$") or before:match("%f[%w_]u8$")
      if char ~= "'" or character_prefix or not text:sub(i - 1, i - 1):match("[%w_]") then
        local cursor = i + 1
        while cursor <= #text do
          local current = text:sub(cursor, cursor)
          if current == "\\" then
            cursor = cursor + 2
          elseif current == char then
            cursor = cursor + 1
            break
          else
            cursor = cursor + 1
          end
        end
        finish = math.min(cursor - 1, #text)
      end
    end
    -- Raw strings can contain arbitrary braces, quotes, and comment-like text.
    if text:sub(i, i + 1) == 'R"' then
      local delimiter = text:match('^R"([^%s%(%)\\]*)%(', i)
      if delimiter and #delimiter <= 16 then
        local closing = ")" .. delimiter .. '"'
        local last = text:find(closing, i + #delimiter + 3, true)
        finish = last and last + #closing - 1 or #text
      end
    end
    if finish then
      local chunk = text:sub(i, finish)
      local hidden = chunk:gsub("[^\n]", " ")
      code[#code + 1] = comment and hidden or chunk
      structure[#structure + 1] = hidden
      i = finish + 1
    else
      code[#code + 1] = char
      structure[#structure + 1] = char
      i = i + 1
    end
  end
  return table.concat(code), table.concat(structure)
end

local function closing_delimiter(text, first, opening, closing)
  local depth = 0
  for i = first, #text do
    local char = text:sub(i, i)
    if char == opening then
      depth = depth + 1
    elseif char == closing then
      depth = depth - 1
      if depth == 0 then
        return i
      end
    end
  end
end

function M.function_status(text, name, id)
  local code, structure = mask_source(text)
  local position = 1
  while position <= #structure do
    local first, last = structure:find("%f[%w_]" .. name .. "%s*%(", position)
    if not first then
      break
    end
    local params_end = closing_delimiter(structure, last, "(", ")")
    local body_start = params_end and structure:find("%S", params_end + 1)
    -- These assignment functions have no trailing qualifiers after their parameters.
    -- Ignore declarations, calls, and the explicit instantiation macro block.
    if body_start and structure:sub(body_start, body_start) == "{" then
      local body_end = closing_delimiter(structure, body_start, "{", "}")
      if not body_end then
        return "work", first, "Unclosed function body"
      end
      local body = code:sub(body_start + 1, body_end - 1)
      local escaped_id = id:gsub("%.", "%%.")
      local stub = 'throw%s+std::logic_error%s*%(%s*"' .. escaped_id .. '[^"\n]*"%s*%)%s*;'
      local remaining, count, cursor = {}, 0, 1
      while cursor <= #body do
        local start, finish = body:find(stub, cursor)
        if not start then
          remaining[#remaining + 1] = body:sub(cursor)
          break
        end
        remaining[#remaining + 1] = body:sub(cursor, start - 1)
        if structure:sub(body_start + start, body_start + start + 4) == "throw" then
          count = count + 1
        else
          remaining[#remaining + 1] = body:sub(start, finish)
        end
        cursor = finish + 1
      end
      local without_stub = table.concat(remaining)
      local status
      if count > 0 then
        status = without_stub:match("%S") and "work" or "todo"
      else
        status = body:match("%S") and "done" or "todo"
      end
      return status, first
    end
    position = last + 1
  end
  return "missing", 1, "Function definition not found"
end

local function read_source(file)
  -- Include unsaved edits when manually refreshing or reopening the sidebar.
  local canonical = vim.fn.resolve(vim.fn.fnamemodify(file, ":p"))
  for _, buf in ipairs(vim.api.nvim_list_bufs()) do
    local name = vim.api.nvim_buf_get_name(buf)
    if vim.api.nvim_buf_is_loaded(buf) and name ~= ""
        and vim.fn.resolve(vim.fn.fnamemodify(name, ":p")) == canonical then
      return table.concat(vim.api.nvim_buf_get_lines(buf, 0, -1, false), "\n")
    end
  end
  local ok, lines = pcall(vim.fn.readfile, file)
  if ok then
    return table.concat(lines, "\n")
  end
end

function M.project(root)
  root = vim.fn.resolve(vim.fn.fnamemodify(root, ":p")):gsub("/$", "")
  local guide = root .. "/docs/solver/implementation-order.md"
  local content = read_source(guide)
  if not content then
    error("Implementation guide not found: " .. guide)
  end
  local result = { root = root, guide = guide, tasks = {}, done = 0, work = 0, todo = 0, missing = 0 }
  local sources = {}
  local line_number = 0
  for line in (content .. "\n"):gmatch("(.-)\n") do
    line_number = line_number + 1
    local order, qualified, id = line:match("^|%s*(%d+)%s*|%s*`([%w_:]+)`%s*/%s*(W%d+%.%d+)%s*|")
    if order then
      local stage, name = qualified:match("^([%w_]+)::([%w_]+)$")
      if stage ~= "scalar" and stage ~= "supernodal" and stage ~= "mt" then
        error("Unknown solver stage in guide: " .. qualified)
      end
      local file = root .. "/src/solver/" .. stage .. "/Lu.cpp"
      if sources[file] == nil then
        sources[file] = read_source(file) or false
      end
      local status, offset, reason = "missing", 1, "Source file not found"
      if sources[file] then
        status, offset, reason = M.function_status(sources[file], name, id)
      end
      local _, newlines = (sources[file] or ""):sub(1, offset - 1):gsub("\n", "")
      result.tasks[#result.tasks + 1] = {
        order = tonumber(order), name = qualified, id = id, stage = stage, file = file,
        line = newlines + 1, guide_line = line_number, status = status, reason = reason,
      }
      result[status] = result[status] + 1
    end
  end
  if #result.tasks == 0 then
    error("No assignment rows found in " .. guide)
  end
  table.sort(result.tasks, function(a, b) return a.order < b.order end)
  for _, task in ipairs(result.tasks) do
    if task.status ~= "done" then
      result.next_task = task
      break
    end
  end
  return result
end

return M
