set(input_dir "${CMAKE_CURRENT_SOURCE_DIR}/inputs")
foreach(ranks IN ITEMS 1 4)
  foreach(input_case IN ITEMS separate_b default_b positional)
    if(input_case STREQUAL "separate_b")
      set(input_args --matrix-a "${input_dir}/a.mtx" --matrix-b "${input_dir}/b.mtx")
    elseif(input_case STREQUAL "default_b")
      set(input_args --matrix-a "${input_dir}/square.mtx")
    else()
      set(input_args "${input_dir}/a.mtx" "${input_dir}/b.mtx")
    endif()
    set(test_name benchmark_summa_inputs_${input_case}_${ranks}_ranks)
    add_test(NAME ${test_name}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:Bench_Mpi2DSummaSpGEMM> ${MPIEXEC_POSTFLAGS} ${input_args})
    set_tests_properties(${test_name} PROPERTIES
      PROCESSORS ${ranks} TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1"
      PASS_REGULAR_EXPRESSION "2D SUMMA CSC check passed")
  endforeach()
endforeach()
