# Run by the native preset at the end of project(SPCraft). The database itself
# is written during generation; a symbolic link can be created beforehand.
set(_spcraft_clangd_dir "${CMAKE_SOURCE_DIR}/build")
if(CMAKE_BINARY_DIR STREQUAL _spcraft_clangd_dir)
  return()
endif()

set(_spcraft_clangd_database "${_spcraft_clangd_dir}/compile_commands.json")
file(RELATIVE_PATH _spcraft_clangd_target "${_spcraft_clangd_dir}"
  "${CMAKE_BINARY_DIR}/compile_commands.json")

if(IS_SYMLINK "${_spcraft_clangd_database}")
  file(READ_SYMLINK "${_spcraft_clangd_database}" _spcraft_clangd_existing)
  if(_spcraft_clangd_existing STREQUAL _spcraft_clangd_target)
    return()
  endif()
endif()

# Another build or the Linux/CUDA parsing setup may own the active database.
if(EXISTS "${_spcraft_clangd_database}" OR IS_SYMLINK "${_spcraft_clangd_database}")
  message(WARNING
    "Preserving the existing clangd database at ${_spcraft_clangd_database}. "
    "Move it aside to activate this preset's database at "
    "${CMAKE_BINARY_DIR}/compile_commands.json.")
  return()
endif()

file(MAKE_DIRECTORY "${_spcraft_clangd_dir}")
file(CREATE_LINK "${_spcraft_clangd_target}" "${_spcraft_clangd_database}" SYMBOLIC
  RESULT _spcraft_clangd_link_result)
if(NOT _spcraft_clangd_link_result STREQUAL "0")
  message(FATAL_ERROR "Could not expose compile commands for clangd: ${_spcraft_clangd_link_result}")
endif()
message(STATUS "clangd database: ${_spcraft_clangd_database} -> ${_spcraft_clangd_target}")
