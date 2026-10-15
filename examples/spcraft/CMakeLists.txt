
add_executable(e01_CooMatrix e01_CooMatrix.cpp)
target_link_libraries(e01_CooMatrix
	PRIVATE
		${EIGEN3_TARGET}
		SPCraft
		cxxopts::cxxopts
		fmt::fmt
)

add_executable(e02_PageRank e02_PageRank.cpp)
target_link_libraries(e02_PageRank
	PRIVATE
		SPCraft
		cxxopts::cxxopts
		fmt::fmt
)

add_executable(e03_column_spgemm e03_column_spgemm.cpp)
target_link_libraries(e03_column_spgemm PRIVATE SPCraft)

add_executable(e04_conjugate_gradient e04_conjugate_gradient.cpp)
target_link_libraries(e04_conjugate_gradient
	PRIVATE
		SPCraft
		cxxopts::cxxopts
		fmt::fmt
)

add_executable(e05_print_matrix e05_print_matrix.cpp)
target_link_libraries(e05_print_matrix PRIVATE SPCraft ${EIGEN3_TARGET} cxxopts::cxxopts)

if(SPCRAFT_USE_MPI)
	add_executable(e06_comm_grid_2d e06_comm_grid_2d.cpp)
	target_link_libraries(e06_comm_grid_2d PRIVATE SPCraft cxxopts::cxxopts)
	add_executable(e07_distributed_sparse_matrix e07_distributed_sparse_matrix.cpp)
	target_link_libraries(e07_distributed_sparse_matrix PRIVATE SPCraft SPCraftInput cxxopts::cxxopts)
endif()

if(SPCRAFT_USE_MPI)
	foreach(example IN ITEMS e08_summa_2d e09_summa_3d)
		add_executable(${example} ${example}.cpp)
		target_link_libraries(${example} PRIVATE SPCraft SPCraftInput ${EIGEN3_TARGET} cxxopts::cxxopts)
	endforeach()
endif()
