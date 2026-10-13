add_library(SPCraftMKLOptions STATIC MKLOptions.cpp)
target_include_directories(SPCraftMKLOptions PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SPCraftMKLOptions PUBLIC SPCraftBenchmarkOptions
	PRIVATE cxxopts::cxxopts MKL::MKL OpenMP::OpenMP_CXX)

add_executable(bench_mkl_spmv MKLSpMVBench.cpp)
target_link_libraries(bench_mkl_spmv PRIVATE SPCraftGoogleBenchmark SPCraftMKLOptions MKL::MKL)

if(BUILD_TESTING)
	add_executable(test_mkl_options ../../tests/test_mkl_options.cpp)
	target_link_libraries(test_mkl_options PRIVATE SPCraftMKLOptions SPCraftTestMain)
	gtest_discover_tests(test_mkl_options TEST_PREFIX mkl_options. DISCOVERY_MODE PRE_TEST)
endif()
