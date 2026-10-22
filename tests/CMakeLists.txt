include(${PROJECT_SOURCE_DIR}/cmake/googletest.cmake)
find_googletest()
include(GoogleTest)

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(kernel)
add_subdirectory(numeric)
add_subdirectory(graph)
add_subdirectory(linear_algebra)
add_subdirectory(utils)
add_subdirectory(semiring)
add_subdirectory(matrix_generator)
add_subdirectory(report)
add_subdirectory(solver)

# Benchmark helper tests are registered by benchmarks/ after their dependencies exist.

# BEGIN scaffold: directory mpi
add_subdirectory(mpi)
# END scaffold: directory mpi
