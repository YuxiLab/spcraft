include(${PROJECT_SOURCE_DIR}/cmake/googletest.cmake)
find_googletest()
include(GoogleTest)

add_library(SPCraftTestMain STATIC TestMain.cpp)
target_link_libraries(SPCraftTestMain PUBLIC GTest::gtest)

# One executable per class or kernel, with named GoogleTest cases.
add_executable(test_csr_matrix test_csr_matrix.cpp)
target_link_libraries(test_csr_matrix PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_csr_matrix TEST_PREFIX csr_matrix. DISCOVERY_MODE PRE_TEST)

add_executable(test_coo_matrix test_coo_matrix.cpp)
target_link_libraries(test_coo_matrix PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_coo_matrix TEST_PREFIX coo_matrix. DISCOVERY_MODE PRE_TEST)

add_executable(test_csc_matrix test_csc_matrix.cpp)
target_link_libraries(test_csc_matrix PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_csc_matrix TEST_PREFIX csc_matrix. DISCOVERY_MODE PRE_TEST)

add_executable(test_dcsc_matrix test_dcsc_matrix.cpp)
target_link_libraries(test_dcsc_matrix PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_dcsc_matrix TEST_PREFIX dcsc_matrix. DISCOVERY_MODE PRE_TEST)

add_executable(test_dense_vector test_dense_vector.cpp)
target_link_libraries(test_dense_vector PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_dense_vector TEST_PREFIX dense_vector. DISCOVERY_MODE PRE_TEST)

add_executable(test_matrix_input_options test_matrix_input_options.cpp)
target_link_libraries(test_matrix_input_options PRIVATE SPCraftTestMain SPCraftInput cxxopts::cxxopts)
gtest_discover_tests(test_matrix_input_options TEST_PREFIX matrix_input_options. DISCOVERY_MODE PRE_TEST)

add_executable(test_print test_print.cpp)
target_link_libraries(test_print PRIVATE SPCraftTestMain SPCraft ${EIGEN3_TARGET})
gtest_discover_tests(test_print TEST_PREFIX print_vector. DISCOVERY_MODE PRE_TEST)

add_executable(test_print_matrix test_print_matrix.cpp)
target_link_libraries(test_print_matrix PRIVATE SPCraftTestMain SPCraft ${EIGEN3_TARGET})
gtest_discover_tests(test_print_matrix TEST_PREFIX print_matrix. DISCOVERY_MODE PRE_TEST)

add_executable(test_mtspmv test_mtspmv.cpp)
target_link_libraries(test_mtspmv PRIVATE SPCraftTestMain ${EIGEN3_TARGET} SPCraft)

gtest_discover_tests(test_mtspmv TEST_PREFIX mtspmv_openmp. DISCOVERY_MODE PRE_TEST)

add_executable(test_mtspgemm test_mtspgemm.cpp)
target_link_libraries(test_mtspgemm PRIVATE SPCraftTestMain ${EIGEN3_TARGET} SPCraft)
gtest_discover_tests(test_mtspgemm TEST_PREFIX omp_hash_spgemm. DISCOVERY_MODE PRE_TEST)

add_executable(test_hash_spgemm_csc test_hash_spgemm_csc.cpp)
target_link_libraries(test_hash_spgemm_csc PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_hash_spgemm_csc TEST_PREFIX hash_spgemm_csc. DISCOVERY_MODE PRE_TEST)

add_executable(test_semiring test_semiring.cpp)
target_link_libraries(test_semiring PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_semiring TEST_PREFIX semiring. DISCOVERY_MODE PRE_TEST)

add_executable(test_matrix_generator test_matrix_generator.cpp)
target_link_libraries(test_matrix_generator PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_matrix_generator TEST_PREFIX matrix_generator. DISCOVERY_MODE PRE_TEST)

add_executable(test_benchmark_report test_benchmark_report.cpp)
target_link_libraries(test_benchmark_report PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_benchmark_report TEST_PREFIX benchmark_report. DISCOVERY_MODE PRE_TEST)

add_executable(test_pagerank test_pagerank.cpp)
target_link_libraries(test_pagerank PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_pagerank TEST_PREFIX pagerank_openmp. DISCOVERY_MODE PRE_TEST)

add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg PRIVATE SPCraftTestMain ${EIGEN3_TARGET} SPCraft)
gtest_discover_tests(test_cg TEST_PREFIX cg_openmp. DISCOVERY_MODE PRE_TEST)

add_executable(test_power_iteration test_power_iteration.cpp)
target_link_libraries(test_power_iteration PRIVATE SPCraftTestMain ${EIGEN3_TARGET} SPCraft)
gtest_discover_tests(test_power_iteration TEST_PREFIX power_iteration_openmp. DISCOVERY_MODE PRE_TEST)

if(SPCRAFT_USE_MKL)
	add_executable(test_mklspmv test_mklspmv.cpp)
	target_link_libraries(test_mklspmv PRIVATE SPCraftTestMain ${EIGEN3_TARGET} SPCraft)
	gtest_discover_tests(test_mklspmv TEST_PREFIX mklspmv. DISCOVERY_MODE PRE_TEST)
endif()

if(SPCRAFT_USE_CUDA)
	add_executable(test_cucsr_matrix test_cucsr_matrix.cu)
	target_link_libraries(test_cucsr_matrix PRIVATE SPCraft)
	add_test(NAME cucsr_matrix COMMAND test_cucsr_matrix)

	add_executable(test_cucsc_matrix test_cucsc_matrix.cu)
	target_link_libraries(test_cucsc_matrix PRIVATE SPCraft)
	add_test(NAME cucsc_matrix COMMAND test_cucsc_matrix)

	add_executable(test_cucoo_matrix test_cucoo_matrix.cu)
	target_link_libraries(test_cucoo_matrix PRIVATE SPCraft)
	add_test(NAME cucoo_matrix COMMAND test_cucoo_matrix)

	add_executable(test_cuspmv test_cuspmv.cu)
	target_link_libraries(test_cuspmv PRIVATE ${EIGEN3_TARGET} SPCraft)
	add_test(NAME cuspmv_block_per_row COMMAND test_cuspmv)

	add_executable(test_cupagerank test_cupagerank.cu)
	target_link_libraries(test_cupagerank PRIVATE SPCraft)
	add_test(NAME pagerank_cuda COMMAND test_cupagerank)
endif()

if(SPCRAFT_USE_COMBBLAS)
	if(NOT TARGET CombBLAS::CombBLAS)
		if(DEFINED CombBLAS_DIR AND NOT IS_ABSOLUTE "${CombBLAS_DIR}")
			get_filename_component(CombBLAS_DIR "${CombBLAS_DIR}" ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")
		endif()
		find_package(CombBLAS CONFIG REQUIRED)
	endif()
	target_link_libraries(test_mtspgemm PRIVATE SPCraftTestMain CombBLAS::CombBLAS)
	target_compile_definitions(test_mtspgemm PRIVATE SPCRAFT_TEST_COMBBLAS=1)
endif()
