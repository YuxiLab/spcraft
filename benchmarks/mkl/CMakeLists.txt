add_executable(bench_mkl_spmv MKLSpMVBench.cpp)
target_link_libraries(bench_mkl_spmv PRIVATE SPCraftBenchmarkCommon MKL::MKL)
