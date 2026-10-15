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

if(SPCRAFT_USE_MPI)
	foreach(format IN ITEMS COO CSR CSC DCSC)
		set(target test_mpi_DistSp${format})
		add_executable(${target} test_DistSp${format}.cpp)
		target_link_libraries(${target} PRIVATE SPCraft GTest::gtest)
		foreach(ranks IN ITEMS 1 4 6)
			add_test(NAME ${target}_${ranks}_ranks
				COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
				${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS})
			set_tests_properties(${target}_${ranks}_ranks
				PROPERTIES PROCESSORS ${ranks} TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
		endforeach()
	endforeach()
endif()

if(SPCRAFT_USE_MPI)
	add_executable(test_mpi_CommGrid2D test_CommGrid2D.cpp)
	target_link_libraries(test_mpi_CommGrid2D PRIVATE SPCraft GTest::gtest)
	foreach(ranks IN ITEMS 1 4 6)
		add_test(NAME test_mpi_CommGrid2D_${ranks}_ranks
			COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
			${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_CommGrid2D> ${MPIEXEC_POSTFLAGS})
		set_tests_properties(test_mpi_CommGrid2D_${ranks}_ranks
			PROPERTIES PROCESSORS ${ranks} TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
	endforeach()
endif()

if(SPCRAFT_USE_MPI)
	add_executable(test_mpi_3dspgemm test_3dspgemm.cpp)
	target_link_libraries(test_mpi_3dspgemm PRIVATE SPCraft GTest::gtest ${EIGEN3_TARGET})
	foreach(ranks IN ITEMS 1 4 8)
		add_test(NAME test_mpi_3dspgemm_${ranks}_ranks
			COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
			${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_3dspgemm> ${MPIEXEC_POSTFLAGS})
		set_tests_properties(test_mpi_3dspgemm_${ranks}_ranks
			PROPERTIES PROCESSORS ${ranks} TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
	endforeach()
endif()

if(SPCRAFT_USE_MPI)
	add_executable(test_mpi_CommGrid3D test_CommGrid3D.cpp)
	target_link_libraries(test_mpi_CommGrid3D PRIVATE SPCraft GTest::gtest)
	foreach(ranks IN ITEMS 1 4 8)
		add_test(NAME test_mpi_CommGrid3D_${ranks}_ranks
			COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
			${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_CommGrid3D> ${MPIEXEC_POSTFLAGS})
		set_tests_properties(test_mpi_CommGrid3D_${ranks}_ranks
			PROPERTIES PROCESSORS ${ranks} TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
	endforeach()
endif()

if(SPCRAFT_USE_MPI)
	add_executable(test_mpi_mpiutil test_mpiutil.cpp)
	target_link_libraries(test_mpi_mpiutil PRIVATE SPCraft GTest::gtest)
	foreach(ranks IN ITEMS 1 4)
		add_test(NAME test_mpi_mpiutil_${ranks}_ranks
			COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
			${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_mpiutil> ${MPIEXEC_POSTFLAGS})
		set_tests_properties(test_mpi_mpiutil_${ranks}_ranks
			PROPERTIES PROCESSORS ${ranks} TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
	endforeach()
endif()
