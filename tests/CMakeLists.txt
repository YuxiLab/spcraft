# One executable per class: the container ownership contract.
add_executable(test_csr_matrix test_csr_matrix.cpp)
target_link_libraries(test_csr_matrix PRIVATE SPCraft)
add_test(NAME csr_matrix COMMAND test_csr_matrix)

add_executable(test_coo_matrix test_coo_matrix.cpp)
target_link_libraries(test_coo_matrix PRIVATE SPCraft)
add_test(NAME coo_matrix COMMAND test_coo_matrix)

add_executable(test_csc_matrix test_csc_matrix.cpp)
target_link_libraries(test_csc_matrix PRIVATE SPCraft)
add_test(NAME csc_matrix COMMAND test_csc_matrix)

add_executable(test_dcsc_matrix test_dcsc_matrix.cpp)
target_link_libraries(test_dcsc_matrix PRIVATE SPCraft)
add_test(NAME dcsc_matrix COMMAND test_dcsc_matrix)

add_executable(test_dense_vector test_dense_vector.cpp)
target_link_libraries(test_dense_vector PRIVATE SPCraft)
add_test(NAME dense_vector COMMAND test_dense_vector)

add_executable(test_mtspmv test_mtspmv.cpp)
target_link_libraries(test_mtspmv PRIVATE ${EIGEN3_TARGET} SPCraft)

add_test(NAME mtspmv_openmp COMMAND test_mtspmv)

add_executable(test_mtspgemm test_mtspgemm.cpp)
target_link_libraries(test_mtspgemm PRIVATE ${EIGEN3_TARGET} SPCraft)
add_test(NAME omp_hash_spgemm COMMAND test_mtspgemm)

add_executable(test_semiring test_semiring.cpp)
target_link_libraries(test_semiring PRIVATE SPCraft)
add_test(NAME semiring COMMAND test_semiring)

add_executable(test_matrix_generator test_matrix_generator.cpp)
target_link_libraries(test_matrix_generator PRIVATE SPCraft)
add_test(NAME matrix_generator COMMAND test_matrix_generator)

add_executable(test_pagerank test_pagerank.cpp)
target_link_libraries(test_pagerank PRIVATE SPCraft)
add_test(NAME pagerank_openmp COMMAND test_pagerank)

add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg PRIVATE ${EIGEN3_TARGET} SPCraft)
add_test(NAME cg_openmp COMMAND test_cg)

add_executable(test_power_iteration test_power_iteration.cpp)
target_link_libraries(test_power_iteration PRIVATE ${EIGEN3_TARGET} SPCraft)
add_test(NAME power_iteration_openmp COMMAND test_power_iteration)

if(SPCRAFT_USE_MKL)
	add_executable(test_mklspmv test_mklspmv.cpp)
	target_link_libraries(test_mklspmv PRIVATE ${EIGEN3_TARGET} SPCraft)
	add_test(NAME mklspmv COMMAND test_mklspmv)
endif()

if(SPCRAFT_USE_CUDA)
	add_executable(test_cucsr_matrix test_cucsr_matrix.cu)
	target_link_libraries(test_cucsr_matrix PRIVATE SPCraft)
	add_test(NAME cucsr_matrix COMMAND test_cucsr_matrix)

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
	target_link_libraries(test_mtspgemm PRIVATE CombBLAS::CombBLAS)
	target_compile_definitions(test_mtspgemm PRIVATE SPCRAFT_TEST_COMBBLAS=1)
endif()
