add_executable(cusparse_spmv_benchmark SpMVBenchmark.cu)
target_link_libraries(cusparse_spmv_benchmark
	PRIVATE
		SPCraftBenchmarkCommon
		CUDA::cudart
		CUDA::cusparse
)
# MPI can propagate GCC LTO flags into CUDA objects. Link their native code
# to avoid LTO discarding NVCC's embedded fatbinary symbols.
target_link_options(cusparse_spmv_benchmark PRIVATE
	$<$<CXX_COMPILER_ID:GNU>:-fno-lto>
)
