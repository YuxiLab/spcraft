cmake_minimum_required(VERSION 3.20)
project(SPCraft VERSION 0.0.1 LANGUAGES C CXX)
include(GNUInstallDirs)

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
option(SPCRAFT_BUILD_DOWNSTREAM "Build tests, examples, benchmarks and Python bindings" OFF)
option(SPCRAFT_BUILD_EXAMPLES "Build SPCraft examples" ${PROJECT_IS_TOP_LEVEL})
option(SPCRAFT_BUILD_PYTHON "Build the optional Python bindings" OFF)
option(SPCRAFT_BUILD_BENCHMARKS "Build SPCraft benchmarks" OFF)
option(SPCRAFT_FETCH_DEPENDENCIES "Download dependencies missing from the installation" ON)
option(SPCRAFT_INSTALL "Install the library and CMake package metadata" OFF)
if(SPCRAFT_INSTALL AND SPCRAFT_FETCH_DEPENDENCIES)
	message(FATAL_ERROR "Installation requires SPCRAFT_FETCH_DEPENDENCIES=OFF and installed dependencies")
endif()


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
# Pick up new host implementation files while kernels migrate out of headers.
file(GLOB_RECURSE SPCRAFT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM SPCRAFT_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/MatrixInput.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/MatrixInputOptions.cpp"
)
add_library(SPCraft SHARED ${SPCRAFT_SOURCES})
target_include_directories(SPCraft
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(SPCraft PUBLIC cxx_std_20)
target_link_libraries(SPCraft
	PUBLIC
		$<BUILD_INTERFACE:Boost::preprocessor>
		$<INSTALL_INTERFACE:Boost::headers>
		$<BUILD_INTERFACE:fast_matrix_market::fast_matrix_market>
		$<INSTALL_INTERFACE:fast_matrix_market::fast_matrix_market>
		$<BUILD_INTERFACE:fmt::fmt>
		$<INSTALL_INTERFACE:fmt::fmt>
		$<BUILD_INTERFACE:${EIGEN3_TARGET}>
		$<INSTALL_INTERFACE:Eigen3::Eigen>
)
if(OpenMP_CXX_FOUND)
	target_link_libraries(SPCraft PUBLIC OpenMP::OpenMP_CXX)
endif()
if(SPCRAFT_USE_MPI)
	target_compile_definitions(SPCraft PUBLIC SPCRAFT_USE_MPI=1)
	target_link_libraries(SPCraft PUBLIC MPI::MPI_CXX)
endif()
if(SPCRAFT_USE_MKL)
	target_compile_definitions(SPCraft PUBLIC SPCRAFT_USE_MKL=1)
	target_link_libraries(SPCraft PUBLIC MKL::MKL)
endif()
if(SPCRAFT_USE_CUDA)
	target_compile_definitions(SPCraft PUBLIC SPCRAFT_USE_CUDA=1)
	add_library(SPCraftCUDA STATIC
    src/cuda/cuSpMV.cu
    src/cuda/cuPageRank.cu
    src/cuda/cuda_error.cu
  )
	target_include_directories(SPCraftCUDA
		PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
			$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	)
	target_compile_definitions(SPCraftCUDA PUBLIC SPCRAFT_USE_CUDA=1)
	target_link_libraries(SPCraftCUDA PUBLIC CUDA::cudart
		$<BUILD_INTERFACE:Boost::preprocessor>
		$<INSTALL_INTERFACE:Boost::headers>
		$<BUILD_INTERFACE:fmt::fmt>
		$<INSTALL_INTERFACE:fmt::fmt>
	)
	target_link_libraries(SPCraft PUBLIC SPCraftCUDA)
endif()

#################################################################
# Build Binaries
#################################################################
# Optional compiled input utilities remain separate from the kernel library.
# EXCLUDE_FROM_ALL avoids compiling the reader unless a consumer uses it.
add_library(SPCraftInput STATIC EXCLUDE_FROM_ALL
  src/utils/MatrixInput.cpp
  src/utils/MatrixInputOptions.cpp
)
target_link_libraries(SPCraftInput PUBLIC SPCraft PRIVATE
	$<BUILD_INTERFACE:cxxopts::cxxopts>
	$<INSTALL_INTERFACE:cxxopts::cxxopts>
)

if(SPCRAFT_BUILD_DOWNSTREAM)
	include(CTest)
	if(BUILD_TESTING)
		add_subdirectory(tests)
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
endif()

# Keep the kernel examples available as explicit targets during library work.
# EXCLUDE_FROM_ALL preserves the library-only default build.
if(SPCRAFT_BUILD_EXAMPLES AND NOT SPCRAFT_BUILD_DOWNSTREAM)
  add_subdirectory(examples/kernel EXCLUDE_FROM_ALL)
endif()

#################################################################
# Install Binaries
#################################################################
if(SPCRAFT_INSTALL)
	# Installation must build the archive exported by SPCraftInput.
	set_target_properties(SPCraftInput PROPERTIES EXCLUDE_FROM_ALL FALSE)
	include(CMakePackageConfigHelpers)
	set(_spcraft_install_targets SPCraft SPCraftInput)
	if(SPCRAFT_USE_CUDA)
		list(APPEND _spcraft_install_targets SPCraftCUDA)
	endif()
	install(TARGETS ${_spcraft_install_targets} EXPORT SPCraftTargets
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	)
	install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
	set(_spcraft_config_dir ${CMAKE_INSTALL_LIBDIR}/cmake/SPCraft)
	configure_package_config_file(cmake/SPCraftConfig.cmake.in
		${CMAKE_CURRENT_BINARY_DIR}/SPCraftConfig.cmake
		INSTALL_DESTINATION ${_spcraft_config_dir}
	)
	write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/SPCraftConfigVersion.cmake
		VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
	)
	install(EXPORT SPCraftTargets NAMESPACE SPCraft:: DESTINATION ${_spcraft_config_dir})
	install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/SPCraftConfig.cmake
		${CMAKE_CURRENT_BINARY_DIR}/SPCraftConfigVersion.cmake
		DESTINATION ${_spcraft_config_dir}
	)
endif()
