macro(find_cuda)
# cuda arch
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL ""
		AND NOT CMAKE_CROSSCOMPILING AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
	set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build")
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_EXTENSIONS OFF)
# for nvhpc 
file(REAL_PATH "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}" _spcraft_cuda_root)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" _spcraft_cuda_version "${CMAKE_CUDA_COMPILER_VERSION}")
set(_spcraft_cuda_math "${_spcraft_cuda_root}/../../math_libs/${_spcraft_cuda_version}")
if(IS_DIRECTORY "${_spcraft_cuda_math}")
	list(PREPEND CMAKE_PREFIX_PATH "${_spcraft_cuda_math}")
endif()
find_package(CUDAToolkit REQUIRED)
if(IS_DIRECTORY "${_spcraft_cuda_math}")
	list(REMOVE_AT CMAKE_PREFIX_PATH 0)
endif()
unset(_spcraft_cuda_root)
unset(_spcraft_cuda_version)
unset(_spcraft_cuda_math)
# find components
foreach(_component IN ITEMS cudart cublas cusolver cusparse)
	if(NOT TARGET CUDA::${_component})
		message(FATAL_ERROR
			"Required CUDA library CUDA::${_component} was not found. "
			"Set CUDAToolkit_ROOT to a complete CUDA toolkit, or add the "
			"matching NVHPC math_libs/<version> directory to CMAKE_PREFIX_PATH.")
	endif()
endforeach()
unset(_component)
endmacro()