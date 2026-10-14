add_library(SPCraftMKLOptions STATIC MKLOptions.cpp)
target_include_directories(SPCraftMKLOptions PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SPCraftMKLOptions PUBLIC SPCraftBenchmarkOptions
	PRIVATE cxxopts::cxxopts MKL::MKL OpenMP::OpenMP_CXX)

add_executable(bench_mkl_spmv MKLSpMVBench.cpp)
target_link_libraries(bench_mkl_spmv PRIVATE SPCraftGoogleBenchmark SPCraftMKLOptions MKL::MKL)

if(BUILD_TESTING)
	add_subdirectory(${PROJECT_SOURCE_DIR}/tests/benchmarks/mkl
		${PROJECT_BINARY_DIR}/tests/benchmarks/mkl)
endif()
