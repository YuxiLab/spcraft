find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_spcraft bindings.cpp)
target_link_libraries(_spcraft PRIVATE SPCraft)

install(TARGETS _spcraft LIBRARY DESTINATION spcraft)
