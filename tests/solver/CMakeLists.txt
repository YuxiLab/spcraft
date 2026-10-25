if(NOT SPCRAFT_BUILD_SOLVER_ASSIGNMENTS)
  return()
endif()

foreach(stage IN ITEMS dense scalar supernodal mt)
  add_executable(test_solver_${stage} test_${stage}_lu.cpp)
  target_link_libraries(test_solver_${stage} PRIVATE SPCraftTestMain SPCraft)
endforeach()

target_sources(test_solver_scalar PRIVATE test_scalar_kernels.cpp)
target_sources(test_solver_supernodal PRIVATE test_supernodal_kernels.cpp)

gtest_discover_tests(test_solver_dense TEST_PREFIX solver.dense.
  PROPERTIES LABELS solver-dense TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_solver_scalar TEST_PREFIX solver.scalar.
  PROPERTIES LABELS solver-scalar TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_solver_supernodal TEST_PREFIX solver.supernodal.
  PROPERTIES LABELS solver-supernodal TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_solver_mt TEST_PREFIX solver.mt.
  PROPERTIES LABELS solver-mt TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
