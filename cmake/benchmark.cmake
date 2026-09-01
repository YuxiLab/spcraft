function(find_benchmark)
	message(STATUS "Fetching Google Benchmark via CPM")
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

	if(TARGET benchmark::benchmark)
		set(BENCHMARK_TARGET benchmark::benchmark PARENT_SCOPE)
	elseif(TARGET benchmark)
		add_library(benchmark::benchmark ALIAS benchmark)
		set(BENCHMARK_TARGET benchmark::benchmark PARENT_SCOPE)
	else()
		message(FATAL_ERROR "Google Benchmark target not found after CPM fetch")
	endif()
endfunction()
