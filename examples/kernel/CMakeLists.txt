set(OPENMP_EXAMPLES ex01_spmv)
foreach(example IN LISTS OPENMP_EXAMPLES)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(
    ${example} PRIVATE
    ${EIGEN3_TARGET}
    SPCraft
    cxxopts::cxxopts fmt::fmt)
  target_include_directories(${example} PRIVATE ${PROJECT_SOURCE_DIR}/examples)
endforeach()
