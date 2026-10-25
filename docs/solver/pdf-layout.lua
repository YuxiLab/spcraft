-- Keep related lessons together in the printable foundations chapter.
function Header(header)
  -- Pandoc applies --shift-heading-level-by after this Lua filter.
  -- Keep numbered walkthrough headings with their opening calculation or matrix.
  if header.level == 3 and pandoc.utils.stringify(header.content):match("^%d+%.") then
    return { pandoc.RawBlock("latex", "\\Needspace{12\\baselineskip}"), header }
  end
  if header.level == 2 then
    local title = pandoc.utils.stringify(header.content)
    local number = title:match("^(%d+)%.")
    if number == "2" or number == "4" or number == "6"
        or title:match("^Stage %d") then
      return { pandoc.RawBlock("latex", "\\clearpage\\thispagestyle{fancy}"), header }
    end
  end
end

function Table(tbl)
  if #tbl.colspecs == 4 and #tbl.head.rows > 0 then
    local first = pandoc.utils.stringify(tbl.head.rows[1].cells[1].contents)
    local widths = first == "Stage" and { 0.08, 0.38, 0.20, 0.34 }
        or first == "Order" and { 0.06, 0.30, 0.26, 0.38 }
    if widths then
      local specs = tbl.colspecs
      for i, width in ipairs(widths) do
        specs[i][2] = width
      end
      tbl.colspecs = specs
    end
  end
  return tbl
end

function Para(paragraph)
  if paragraph.content[1] and paragraph.content[1].t == "Str"
      and paragraph.content[1].text == "Exercise:" then
    paragraph.content:remove(1)
    if paragraph.content[1] and paragraph.content[1].t == "Space" then
      paragraph.content:remove(1)
    end
    if paragraph.content[1] and paragraph.content[1].t == "Str" then
      paragraph.content[1].text = paragraph.content[1].text:gsub("^%l", string.upper)
    end
    return {
      pandoc.RawBlock("latex", "\\begin{exercise}"),
      paragraph,
      pandoc.RawBlock("latex", "\\end{exercise}")
    }
  end
end

-- Both docs/pdf (published) and docs/obj (IDE preview) sit beside docs/solver.
function Link(link)
  if not link.target:match("^%a[%w+.-]*:") and not link.target:match("^[#/]") then
    link.target = "../solver/" .. link.target
  end
  return link
end
