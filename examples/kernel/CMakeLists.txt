set(OPENMP_EXAMPLES ex01_spmv ex02_spgemm)
foreach(example IN LISTS OPENMP_EXAMPLES)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(
    ${example} PRIVATE
    SPCraft
    SPCraftInput
    cxxopts::cxxopts fmt::fmt)
  target_precompile_headers(${example} PRIVATE <utils/StablePch.h>)
endforeach()
