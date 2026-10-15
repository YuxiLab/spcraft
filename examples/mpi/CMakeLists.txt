if(SPCRAFT_USE_MPI)
	set(MPI_TEACHING_EXAMPLES
		e01_mpi_hello
		e02_mpi_send_receive
		e03_mpi_collectives
	)
	foreach(example IN LISTS MPI_TEACHING_EXAMPLES)
		add_executable(${example} ${example}.cpp)
		target_link_libraries(${example} PRIVATE MPI::MPI_CXX)
	endforeach()

	add_executable(e04_mpi_hello e04_mpi_hello.cpp)
	target_link_libraries(e04_mpi_hello
		PRIVATE
			${EIGEN3_TARGET}
			SPCraft
			fmt::fmt
	)
else()
	message(STATUS "MPI support is disabled; skipping the MPI example")
endif()
