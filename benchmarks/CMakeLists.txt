# Helpers shared by every benchmark folder: dataset loading, the Eigen
# reference, warmup, and the iteration/time-budget policy.
include(${PROJECT_SOURCE_DIR}/cmake/benchmark.cmake)
find_benchmark()

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

add_library(SPCraftGoogleBenchmark INTERFACE)
target_link_libraries(SPCraftGoogleBenchmark INTERFACE SPCraftBenchmarkCommon benchmark::benchmark)

if(BUILD_TESTING)
	add_subdirectory(${PROJECT_SOURCE_DIR}/tests/benchmarks/common
		${PROJECT_BINARY_DIR}/tests/benchmarks/common)
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

# BEGIN scaffold: directory mpi
add_subdirectory(mpi)
# END scaffold: directory mpi
