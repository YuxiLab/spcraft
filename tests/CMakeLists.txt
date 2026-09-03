add_executable(test_mtspmv test_mtspmv.cpp)
target_link_libraries(test_mtspmv PRIVATE ${EIGEN3_TARGET} SPCraft)

add_test(NAME mtspmv_openmp COMMAND test_mtspmv)

add_executable(test_semiring test_semiring.cpp)
target_link_libraries(test_semiring PRIVATE SPCraft)
add_test(NAME semiring COMMAND test_semiring)

add_executable(test_matrix_generator test_matrix_generator.cpp)
target_link_libraries(test_matrix_generator PRIVATE SPCraft)
add_test(NAME matrix_generator COMMAND test_matrix_generator)

if(SPCRAFT_USE_MKL)
	add_executable(test_mklspmv test_mklspmv.cpp)
	target_link_libraries(test_mklspmv PRIVATE ${EIGEN3_TARGET} SPCraft)
	add_test(NAME mklspmv COMMAND test_mklspmv)
endif()

if(SPCRAFT_USE_CUDA)
	add_executable(test_cuspmv test_cuspmv.cu)
	target_link_libraries(test_cuspmv PRIVATE ${EIGEN3_TARGET} SPCraft)
	add_test(NAME cuspmv_block_per_row COMMAND test_cuspmv)
endif()
