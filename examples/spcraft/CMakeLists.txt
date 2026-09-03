
add_executable(e01_CooMatrix e01_CooMatrix.cpp)
target_link_libraries(e01_CooMatrix
	PRIVATE
		${EIGEN3_TARGET}
		SPCraft
		cxxopts::cxxopts
		fmt::fmt
)
