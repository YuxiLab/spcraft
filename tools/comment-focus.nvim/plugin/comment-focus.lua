if vim.g.loaded_comment_focus then
  return
end
vim.g.loaded_comment_focus = true
require('comment-focus')._load()
