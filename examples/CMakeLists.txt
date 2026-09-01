add_executable(e01_read_csrmatrix e01_read_csrmatrix.cpp)
target_link_libraries(e01_read_csrmatrix
	PRIVATE
		SPCraft
		cxxopts::cxxopts
		fmt::fmt
)

add_executable(e02_compiler_error e02_compiler_error.cpp)
target_link_libraries(e02_compiler_error
	PRIVATE
		SPCraft
		cxxopts::cxxopts
		fmt::fmt
)
