add_executable(test_pagerank test_mtPageRank.cpp)
target_link_libraries(test_pagerank PRIVATE SPCraftTestMain SPCraft)
gtest_discover_tests(test_pagerank TEST_PREFIX pagerank_openmp. DISCOVERY_MODE PRE_TEST)

if(SPCRAFT_USE_CUDA)
	add_executable(test_cupagerank test_cuPageRank.cu)
	target_link_libraries(test_cupagerank PRIVATE SPCraft)
	add_test(NAME pagerank_cuda COMMAND test_cupagerank)
endif()
