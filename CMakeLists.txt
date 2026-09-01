cmake_minimum_required(VERSION 3.20)
project(SPCraft VERSION 0.0.1 LANGUAGES CXX)

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

# Options
option(SPCRAFT_BUILD_EXAMPLES "Build SPCraft examples" ${PROJECT_IS_TOP_LEVEL})
option(SPCRAFT_BUILD_PYTHON "Build the optional Python bindings" OFF)
option(SPCRAFT_BUILD_BENCHMARKS "Build SPCraft benchmarks" OFF)

include(CTest)

# CPM & Dependencies
include(cmake/CPM.cmake)
include(cmake/fast_matrix_market.cmake)
include(cmake/cxxopts.cmake)
include(cmake/fmt.cmake)
include(cmake/benchmark.cmake)

find_fast_matrix_market()
find_cxxopts()
find_fmt()
find_benchmark()
find_package(OpenMP QUIET COMPONENTS CXX)

# SPCraft header-only interface library
add_library(SPCraft INTERFACE)
target_include_directories(SPCraft
	INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(SPCraft INTERFACE fast_matrix_market::fast_matrix_market)
if(OpenMP_CXX_FOUND)
	target_link_libraries(SPCraft INTERFACE OpenMP::OpenMP_CXX)
endif()

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
