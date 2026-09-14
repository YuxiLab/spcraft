cmake_minimum_required(VERSION 3.20)
project(SPCraft VERSION 0.0.1 LANGUAGES C CXX)

#################################################################
# Compiler options
#################################################################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

#################################################################
# Options
#################################################################
option(SPCRAFT_USE_MPI "Activate the MPI dependency" OFF)
option(SPCRAFT_USE_CUDA "Activate the CUDA dependency" OFF)
option(SPCRAFT_USE_MKL "Activate the oneMKL sparse backend" OFF)
option(SPCRAFT_BENCH_COMBBLAS "Activate the CombBLAS comparison benchmarks" OFF)
option(SPCRAFT_BENCH_GINKGO "Activate the GINKGO comparison benchmarks" OFF)
option(SPCRAFT_BENCH_NETWORKX "Activate the GINKGO comparison benchmarks" OFF)
option(SPCRAFT_BENCH_GraphBLAS "Activate the GraphBLAS comparison benchmarks" OFF)
option(SPCRAFT_BUILD_EXAMPLES "Build SPCraft examples" ${PROJECT_IS_TOP_LEVEL})
option(SPCRAFT_BUILD_PYTHON "Build the optional Python bindings" OFF)
option(SPCRAFT_BUILD_BENCHMARKS "Build SPCraft benchmarks" OFF)


#################################################################
# CPM & Dependencies
#################################################################
include(cmake/CPM.cmake)
include(cmake/boost.cmake)
include(cmake/eigen3.cmake)
include(cmake/fast_matrix_market.cmake)
include(cmake/cxxopts.cmake)
include(cmake/fmt.cmake)
include(cmake/cuda.cmake)

find_boost()
find_eigen3()
find_fast_matrix_market()
find_cxxopts()
find_fmt()
find_package(OpenMP QUIET COMPONENTS CXX)

if(SPCRAFT_USE_MKL)
	set(MKL_LINK dynamic)
	set(MKL_THREADING gnu_thread)
	set(MKL_INTERFACE lp64)
	find_package(MKL CONFIG REQUIRED
		PATHS
			"$ENV{MKLROOT}"
			"/opt/intel/oneapi/mkl/latest/lib/cmake/mkl"
	)
endif()

if(SPCRAFT_USE_MPI)
	find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(SPCRAFT_USE_CUDA)
	find_cuda()
endif()

#################################################################
# Cmake Binaries Settings
#################################################################
# Place all executables in <build_dir>/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# (Optional) Place compiled static/shared libraries in <build_dir>/lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#################################################################
# Build SPCraft main library
#################################################################
add_library(SPCraft INTERFACE)
target_include_directories(SPCraft
	INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(SPCraft
	INTERFACE
		Boost::preprocessor
		fast_matrix_market::fast_matrix_market
		fmt::fmt
		${EIGEN3_TARGET}
)
if(OpenMP_CXX_FOUND)
	target_link_libraries(SPCraft INTERFACE OpenMP::OpenMP_CXX)
endif()
if(SPCRAFT_USE_MPI)
	target_compile_definitions(SPCraft INTERFACE SPCRAFT_USE_MPI=1)
	target_link_libraries(SPCraft INTERFACE MPI::MPI_CXX)
endif()
if(SPCRAFT_USE_MKL)
	target_compile_definitions(SPCraft INTERFACE SPCRAFT_USE_MKL=1)
	target_link_libraries(SPCraft INTERFACE MKL::MKL)
endif()
if(SPCRAFT_USE_CUDA)
	target_compile_definitions(SPCraft INTERFACE SPCRAFT_USE_CUDA=1)
	add_library(SPCraftCUDA STATIC
    src/cuda/cuSpMV.cu
    src/cuda/cuPageRank.cu
    src/cuda/cuda_error.cu
  )
	target_include_directories(SPCraftCUDA
		PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
	)
	target_compile_definitions(SPCraftCUDA PUBLIC SPCRAFT_USE_CUDA=1)
	target_link_libraries(SPCraftCUDA PUBLIC CUDA::cudart Boost::preprocessor fmt::fmt)
	target_link_libraries(SPCraft INTERFACE SPCraftCUDA)
endif()

#################################################################
# Build Binaries
#################################################################
include(CTest)
if(BUILD_TESTING)
	if(OpenMP_CXX_FOUND)
		add_subdirectory(tests)
	else()
		message(STATUS "OpenMP was not found; skipping the OpenMP SpMV test")
	endif()
endif()

if(SPCRAFT_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

if(SPCRAFT_BUILD_BENCHMARKS)
	if(NOT OpenMP_CXX_FOUND)
		message(FATAL_ERROR "SPCraft benchmarks require OpenMP")
	endif()
	add_subdirectory(benchmarks)
endif()

if(SPCRAFT_BUILD_PYTHON)
	add_subdirectory(python)
endif()

#################################################################
# Install Binaries
#################################################################
