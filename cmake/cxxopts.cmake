function(find_cxxopts)
	message(STATUS "Fetching cxxopts 3.3.1 via CPM")
	CPMAddPackage(
		NAME cxxopts
		URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.3.1.tar.gz
		VERSION 3.3.1
		OPTIONS
			"CXXOPTS_BUILD_EXAMPLES OFF"
			"CXXOPTS_BUILD_TESTS OFF"
	)

	if(TARGET cxxopts::cxxopts)
		set(CXXOPTS_TARGET cxxopts::cxxopts PARENT_SCOPE)
	else()
		message(FATAL_ERROR "cxxopts target not found after CPM fetch")
	endif()
endfunction()