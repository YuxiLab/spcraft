add_executable(test_csr_matrix test_CsrMatrix.cpp)
target_link_libraries(test_csr_matrix PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_csr_matrix TEST_PREFIX csr_matrix. DISCOVERY_MODE PRE_TEST)

add_executable(test_coo_matrix test_CooMatrix.cpp)
target_link_libraries(test_coo_matrix PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_coo_matrix TEST_PREFIX coo_matrix. DISCOVERY_MODE PRE_TEST)

add_executable(test_csc_matrix test_CscMatrix.cpp)
target_link_libraries(test_csc_matrix PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_csc_matrix TEST_PREFIX csc_matrix. DISCOVERY_MODE PRE_TEST)

add_executable(test_dcsc_matrix test_DcscMatrix.cpp)
target_link_libraries(test_dcsc_matrix PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_dcsc_matrix TEST_PREFIX dcsc_matrix. DISCOVERY_MODE PRE_TEST)

add_executable(test_dense_vector test_DenseVector.cpp)
target_link_libraries(test_dense_vector PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_dense_vector TEST_PREFIX dense_vector. DISCOVERY_MODE PRE_TEST)

if(SPCRAFT_USE_CUDA)
	add_executable(test_cucsr_matrix test_cuCsrMatrix.cu)
	target_link_libraries(test_cucsr_matrix PRIVATE SPCraft)
	add_test(NAME cucsr_matrix COMMAND test_cucsr_matrix)
endif()

if(SPCRAFT_USE_CUDA)
	add_executable(test_cucsc_matrix test_cuCscMatrix.cu)
	target_link_libraries(test_cucsc_matrix PRIVATE SPCraft)
	add_test(NAME cucsc_matrix COMMAND test_cucsc_matrix)
endif()

if(SPCRAFT_USE_CUDA)
	add_executable(test_cucoo_matrix test_cuCooMatrix.cu)
	target_link_libraries(test_cucoo_matrix PRIVATE SPCraft)
	add_test(NAME cucoo_matrix COMMAND test_cucoo_matrix)
endif()
