set(EIGEN_SOLVER_EXAMPLES
    e01_triangular_solve
    e02_lu_pivoting
    e03_conditioning_refinement
    e04_cholesky
    e05_least_squares
    e06_rank_deficient_svd
    e07_dense_vs_sparse
    e08_conjugate_gradient)

foreach(example IN LISTS EIGEN_SOLVER_EXAMPLES)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE ${EIGEN3_TARGET} cxxopts::cxxopts)
endforeach()
