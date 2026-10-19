
add_executable(bench_ompspmv kernels/OmpSpMVBench.cpp)
target_link_libraries(bench_ompspmv PRIVATE SPCraftGoogleBenchmark)

if(SPCRAFT_USE_MPI)
	add_subdirectory(mpi)
endif()
