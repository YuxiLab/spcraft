if(OpenMP_CXX_FOUND)
	add_executable(e03_spmv_openmp e03_spmv_openmp.cpp)
	target_link_libraries(e03_spmv_openmp
		PRIVATE
			SPCraft
			cxxopts::cxxopts
			fmt::fmt
	)
else()
	message(STATUS "OpenMP was not found; skipping the OpenMP example")
endif()
