# BEGIN scaffold: target bench_mpi_2dspgemm
if(SPCRAFT_USE_MPI)
	add_executable(bench_mpi_2dspgemm bench_2dspgemm.cpp)
	target_link_libraries(bench_mpi_2dspgemm PRIVATE SPCraft benchmark::benchmark)
endif()
# END scaffold: target bench_mpi_2dspgemm
