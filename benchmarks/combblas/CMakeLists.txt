if(DEFINED CombBLAS_DIR AND NOT IS_ABSOLUTE "${CombBLAS_DIR}")
	get_filename_component(CombBLAS_DIR "${CombBLAS_DIR}" ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")
endif()
find_package(CombBLAS CONFIG REQUIRED)

add_executable(spgemm_combblas_benchmark SpGEMMBenchmark.cpp)
target_link_libraries(spgemm_combblas_benchmark PRIVATE
	SPCraftBenchmarkCommon CombBLAS::CombBLAS)

# Record the revision supplied by the caller for installed packages, where the
# original checkout need not exist beside the benchmark.
set(SPCRAFT_COMBBLAS_REVISION "unknown" CACHE STRING "CombBLAS revision used by benchmarks")
target_compile_definitions(spgemm_combblas_benchmark PRIVATE
	SPCRAFT_COMBBLAS_REVISION="${SPCRAFT_COMBBLAS_REVISION}"
	SPCRAFT_BENCHMARK_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

# A separate storage-conversion diagnostic uses only the external CombBLAS API.
add_executable(spgemm_conversion_benchmark SpGEMMConversionBenchmark.cpp)
target_link_libraries(spgemm_conversion_benchmark PRIVATE SPCraftBenchmarkCommon CombBLAS::CombBLAS)
