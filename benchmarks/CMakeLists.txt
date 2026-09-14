# Helpers shared by every benchmark folder: dataset loading, the Eigen
# reference, warmup, and the iteration/time-budget policy.
add_library(SPCraftBenchmarkCommon INTERFACE)
target_include_directories(SPCraftBenchmarkCommon INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SPCraftBenchmarkCommon
	INTERFACE
		SPCraft
		${EIGEN3_TARGET}
		OpenMP::OpenMP_CXX
		cxxopts::cxxopts
)

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
