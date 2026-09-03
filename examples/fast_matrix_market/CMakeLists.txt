add_executable(e01_read_matrix_market e01_read_matrix_market.cpp)
target_link_libraries(e01_read_matrix_market
	PRIVATE
		${EIGEN3_TARGET}
		fast_matrix_market::fast_matrix_market
		fmt::fmt
)
