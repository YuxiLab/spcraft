if(OpenMP_CXX_FOUND)
	set(OPENMP_TEACHING_EXAMPLES
		e01_openmp_hello
		e02_openmp_parallel_for
		e03_openmp_reduction
	)
	foreach(example IN LISTS OPENMP_TEACHING_EXAMPLES)
		add_executable(${example} ${example}.cpp)
		target_link_libraries(${example} PRIVATE OpenMP::OpenMP_CXX)
	endforeach()

	add_executable(e01_spmv e01_spmv.cpp)
	target_link_libraries(e01_spmv
		PRIVATE
			${EIGEN3_TARGET}
			SPCraft
			cxxopts::cxxopts
			fmt::fmt
	)
else()
	message(STATUS "OpenMP was not found; skipping the OpenMP example")
endif()
