add_executable(test_mtspmv test_mtspmv.cpp)
target_link_libraries(test_mtspmv PRIVATE SPCraft)

add_test(NAME mtspmv_openmp COMMAND test_mtspmv)

if(SPCRAFT_USE_MKL)
	add_executable(test_mklspmv test_mklspmv.cpp)
	target_link_libraries(test_mklspmv PRIVATE SPCraft)
	add_test(NAME mklspmv COMMAND test_mklspmv)
endif()

if(SPCRAFT_USE_CUDA)
	add_executable(test_cuspmv test_cuspmv.cu)
	target_link_libraries(test_cuspmv PRIVATE SPCraft)
	add_test(NAME cuspmv_block_per_row COMMAND test_cuspmv)
endif()
