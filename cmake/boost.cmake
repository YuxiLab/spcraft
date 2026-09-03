function(find_boost)
	find_package(Boost CONFIG QUIET)
	if(NOT Boost_FOUND)
		if(POLICY CMP0167)
			cmake_policy(PUSH)
			cmake_policy(SET CMP0167 OLD)
		endif()

		find_package(Boost QUIET)

		if(POLICY CMP0167)
			cmake_policy(POP)
		endif()
	endif()

	if(Boost_FOUND)
		message(STATUS "Boost found, using existing installation")

		if(NOT TARGET Boost::preprocessor)
			add_library(spcraft_boost_preprocessor INTERFACE)

			if(TARGET Boost::headers)
				target_link_libraries(spcraft_boost_preprocessor INTERFACE Boost::headers)
			elseif(TARGET Boost::boost)
				target_link_libraries(spcraft_boost_preprocessor INTERFACE Boost::boost)
			elseif(Boost_INCLUDE_DIRS)
				target_include_directories(spcraft_boost_preprocessor
					SYSTEM INTERFACE ${Boost_INCLUDE_DIRS}
				)
			else()
				message(FATAL_ERROR "Boost was found, but its headers are not available")
			endif()

			add_library(Boost::preprocessor ALIAS spcraft_boost_preprocessor)
		endif()
	else()
		message(STATUS "Boost not found, adding Boost.Preprocessor via CPM")
		CPMAddPackage(
			NAME boost_preprocessor
			GITHUB_REPOSITORY boostorg/preprocessor
			GIT_TAG boost-1.91.0
		)
	endif()

	if(NOT TARGET Boost::preprocessor)
		message(FATAL_ERROR "Boost.Preprocessor target not available")
	endif()
endfunction()
