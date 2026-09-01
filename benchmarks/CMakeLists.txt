add_executable(spmv_benchmark spmv_benchmark.cpp)
target_link_libraries(spmv_benchmark
	PRIVATE
		SPCraft
		benchmark::benchmark
		cxxopts::cxxopts
)

add_executable(spmv_er_openmp spmv_er_openmp.cpp)
target_link_libraries(spmv_er_openmp PRIVATE SPCraft)

