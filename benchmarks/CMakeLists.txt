# Helpers shared by every benchmark folder: dataset loading, the Eigen
# reference, warmup, and the iteration/time-budget policy.
add_library(SPCraftBenchmarkOptions STATIC
	common/BenchmarkOptions.cpp
	common/BenchmarkInput.cpp
)
target_include_directories(SPCraftBenchmarkOptions PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SPCraftBenchmarkOptions PRIVATE cxxopts::cxxopts)

add_library(SPCraftBenchmarkCommon INTERFACE)
target_include_directories(SPCraftBenchmarkCommon INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SPCraftBenchmarkCommon
	INTERFACE
		SPCraft
		SPCraftBenchmarkOptions
		${EIGEN3_TARGET}
		OpenMP::OpenMP_CXX
		cxxopts::cxxopts
)

if(BUILD_TESTING)
	add_executable(test_benchmark_options ../tests/test_benchmark_options.cpp)
	target_link_libraries(test_benchmark_options PRIVATE SPCraftBenchmarkOptions cxxopts::cxxopts)
	add_test(NAME benchmark_options COMMAND test_benchmark_options)
endif()

add_subdirectory(spcraft)

if(SPCRAFT_BENCH_COMBBLAS)
	add_subdirectory(combblas)
endif()

if(SPCRAFT_USE_MKL)
	add_subdirectory(mkl)
endif()

if(SPCRAFT_USE_CUDA)
	add_subdirectory(cusparse)
endif()
