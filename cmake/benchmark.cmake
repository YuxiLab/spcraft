function(find_benchmark)
	find_package(benchmark 1.9.1 QUIET CONFIG)
	if(NOT benchmark_FOUND)
		message(STATUS "Google Benchmark not found, adding it via CPM")
		CPMAddPackage(
			NAME benchmark
			GITHUB_REPOSITORY google/benchmark
			GIT_TAG v1.9.1
			OPTIONS
				"BENCHMARK_ENABLE_TESTING OFF"
				"BENCHMARK_ENABLE_INSTALL OFF"
				"BENCHMARK_ENABLE_DOXYGEN OFF"
				"BENCHMARK_INSTALL_DOCS OFF"
		)
	else()
		message(STATUS "Google Benchmark found, using existing installation")
	endif()

	if(TARGET benchmark::benchmark)
		set(BENCHMARK_TARGET benchmark::benchmark PARENT_SCOPE)
	elseif(TARGET benchmark)
		add_library(benchmark::benchmark ALIAS benchmark)
		set(BENCHMARK_TARGET benchmark::benchmark PARENT_SCOPE)
	else()
		message(FATAL_ERROR "Google Benchmark target not available")
	endif()
endfunction()
