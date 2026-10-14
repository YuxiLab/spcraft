function(find_fast_matrix_market)
	find_package(fast_matrix_market QUIET)
	if(NOT fast_matrix_market_FOUND)
		if(NOT SPCRAFT_FETCH_DEPENDENCIES)
			message(FATAL_ERROR "fast_matrix_market was not found; install it or enable SPCRAFT_FETCH_DEPENDENCIES")
		endif()
		message(STATUS "fast_matrix_market not found, adding it via CPM")
		CPMAddPackage(
			NAME fast_matrix_market
			GITHUB_REPOSITORY alugowski/fast_matrix_market
			VERSION 1.7.6
			OPTIONS "FAST_MATRIX_MARKET_TEST OFF"
		)
	else()
		message(STATUS "fast_matrix_market found, using existing installation")
	endif()
endfunction()
