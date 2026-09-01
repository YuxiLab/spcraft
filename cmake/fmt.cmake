function(find_fmt)
	find_package(fmt 10.0.0 QUIET)
	if(NOT fmt_FOUND)
		message(STATUS "fmt not found, adding it via CPM")
		CPMAddPackage(
			NAME fmt
			GITHUB_REPOSITORY fmtlib/fmt
			GIT_TAG 11.0.2
		)
	else()
		message(STATUS "fmt found, using existing installation")
	endif()

	if(NOT TARGET fmt::fmt)
		message(FATAL_ERROR "fmt target not available")
	endif()
endfunction()
