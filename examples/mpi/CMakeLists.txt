if(SPCRAFT_USE_MPI)
	add_executable(e04_mpi_hello e04_mpi_hello.cpp)
	target_link_libraries(e04_mpi_hello
		PRIVATE
			SPCraft
			fmt::fmt
	)
else()
	message(STATUS "MPI support is disabled; skipping the MPI example")
endif()
