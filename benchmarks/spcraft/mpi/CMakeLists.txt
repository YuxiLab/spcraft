# SPCraft MPI benchmarks.

function(spcraft_add_mpi_benchmark target)
	add_executable(${target} ${target}.cpp)
	target_link_libraries(${target} PRIVATE SPCraft SPCraftInput cxxopts::cxxopts)
endfunction()

if(SPCRAFT_USE_MPI)
  spcraft_add_mpi_benchmark(Bench_Mpi2DSummaSpGEMM)
  if(BUILD_TESTING)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests/benchmarks/mpi
      ${PROJECT_BINARY_DIR}/tests/benchmarks/mpi)
  endif()
endif()
