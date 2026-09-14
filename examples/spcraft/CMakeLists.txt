
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
