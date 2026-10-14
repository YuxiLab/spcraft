function(find_googletest)
	find_package(GTest 1.17.0 QUIET CONFIG)
	if(NOT GTest_FOUND)
		if(NOT SPCRAFT_FETCH_DEPENDENCIES)
			message(FATAL_ERROR "GTest was not found; install it or enable SPCRAFT_FETCH_DEPENDENCIES")
		endif()
		message(STATUS "GoogleTest not found, adding it via CPM")
		CPMAddPackage(
			NAME googletest
			URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
			VERSION 1.17.0
			OPTIONS
				"INSTALL_GTEST OFF"
				"BUILD_GMOCK OFF"
		)
	else()
		message(STATUS "GoogleTest found, using existing installation")
	endif()

	if(NOT TARGET GTest::gtest)
		message(FATAL_ERROR "GoogleTest target not available")
	endif()
endfunction()
