# BEGIN scaffold: target test_mpi_2dspgemm
if(SPCRAFT_USE_MPI)
	add_executable(test_mpi_2dspgemm test_2dspgemm.cpp)
	target_link_libraries(test_mpi_2dspgemm PRIVATE SPCraft GTest::gtest ${EIGEN3_TARGET})
	foreach(ranks IN ITEMS 1 4)
		add_test(NAME test_mpi_2dspgemm_${ranks}_ranks
			COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
			${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_2dspgemm> ${MPIEXEC_POSTFLAGS})
		set_tests_properties(test_mpi_2dspgemm_${ranks}_ranks
			PROPERTIES PROCESSORS ${ranks} TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
	endforeach()
endif()
# END scaffold: target test_mpi_2dspgemm
