add_executable(e01_read_matrix_market e01_read_matrix_market.cpp)
target_link_libraries(e01_read_matrix_market
	PRIVATE
		fast_matrix_market::fast_matrix_market
		fmt::fmt
)

add_executable(e02_compiler_error e02_compiler_error.cpp)
target_link_libraries(e02_compiler_error
	PRIVATE
		SPCraft
		cxxopts::cxxopts
		fmt::fmt
)
