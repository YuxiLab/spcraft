if(NOT SPCRAFT_BUILD_SOLVER_ASSIGNMENTS)
  return()
endif()

foreach(stage IN ITEMS scalar supernodal mt)
  add_executable(test_solver_${stage} test_${stage}_lu.cpp)
  target_link_libraries(test_solver_${stage} PRIVATE SPCraftTestMain SPCraft)
endforeach()

gtest_discover_tests(test_solver_scalar TEST_PREFIX solver.scalar.
  PROPERTIES LABELS solver-week1 TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_solver_supernodal TEST_PREFIX solver.supernodal.
  PROPERTIES LABELS solver-week2 TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_solver_mt TEST_PREFIX solver.mt.
  PROPERTIES LABELS solver-week3 TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
