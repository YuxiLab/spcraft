function(find_cxxopts)
	find_package(cxxopts 3.3.1 QUIET CONFIG)
	if(NOT cxxopts_FOUND)
		message(STATUS "cxxopts not found, adding it via CPM")
		CPMAddPackage(
			NAME cxxopts
			URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.3.1.tar.gz
			VERSION 3.3.1
			OPTIONS
				"CXXOPTS_BUILD_EXAMPLES OFF"
				"CXXOPTS_BUILD_TESTS OFF"
		)
	else()
		message(STATUS "cxxopts found, using existing installation")
	endif()

	if(TARGET cxxopts::cxxopts)
		set(CXXOPTS_TARGET cxxopts::cxxopts PARENT_SCOPE)
	else()
		message(FATAL_ERROR "cxxopts target not available")
	endif()
endfunction()
