find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_spcraft bindings.cpp)
target_link_libraries(_spcraft PRIVATE SPCraft)

install(TARGETS _spcraft LIBRARY DESTINATION spcraft)
