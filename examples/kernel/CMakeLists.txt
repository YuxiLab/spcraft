add_library(spcraft_example_eigeninterface STATIC ../eigeninterface.cpp)
target_link_libraries(spcraft_example_eigeninterface PRIVATE ${EIGEN3_TARGET} SPCraft)
target_include_directories(spcraft_example_eigeninterface PUBLIC ${PROJECT_SOURCE_DIR}/examples)

set(OPENMP_EXAMPLES ex01_spmv ex02_spgemm)
foreach(example IN LISTS OPENMP_EXAMPLES)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(
    ${example} PRIVATE
    SPCraft
    spcraft_example_eigeninterface
    SPCraftInput
    cxxopts::cxxopts fmt::fmt)
  target_include_directories(${example} PRIVATE ${PROJECT_SOURCE_DIR}/examples)
  target_precompile_headers(${example} PRIVATE <utils/StablePch.h>)
endforeach()
