add_library(SPCraftMKLOptions STATIC MKLOptions.cpp)
target_include_directories(SPCraftMKLOptions PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SPCraftMKLOptions PUBLIC SPCraftBenchmarkOptions
	PRIVATE MKL::MKL OpenMP::OpenMP_CXX)

add_executable(bench_mkl_spmv MKLSpMVBench.cpp)
target_link_libraries(bench_mkl_spmv PRIVATE SPCraftBenchmarkCommon SPCraftMKLOptions MKL::MKL)

if(BUILD_TESTING)
	add_executable(test_mkl_options ../../tests/test_mkl_options.cpp)
	target_link_libraries(test_mkl_options PRIVATE SPCraftMKLOptions)
	add_test(NAME mkl_options COMMAND test_mkl_options)
endif()
