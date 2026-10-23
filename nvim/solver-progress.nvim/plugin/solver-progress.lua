if vim.g.loaded_solver_progress then
  return
end
vim.g.loaded_solver_progress = true

require("solver_progress").setup()
