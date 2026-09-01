add_executable(test_mtspmv test_mtspmv.cpp)
target_link_libraries(test_mtspmv PRIVATE SPCraft)

add_test(NAME mtspmv_openmp COMMAND test_mtspmv)
