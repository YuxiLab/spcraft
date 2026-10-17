# ProfilerRun accepts one build command; configure and build in sequence here.
cmake_minimum_required(VERSION 3.21)
get_filename_component(project_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(configure_args --preset profiler)
if(DEFINED ENV{PROFILER_TOOLS_DIR})
  list(APPEND configure_args "-DSPCRAFT_PROFILER_TOOLS=$ENV{PROFILER_TOOLS_DIR}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" ${configure_args}
  WORKING_DIRECTORY "${project_dir}"
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build --preset profiler --parallel 4
  WORKING_DIRECTORY "${project_dir}"
  COMMAND_ERROR_IS_FATAL ANY
)
