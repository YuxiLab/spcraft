# BEGIN scaffold: source 2dspgemm
if(SPCRAFT_USE_MPI)
	target_sources(SPCraft PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/2dspgemm.cpp")
endif()
# END scaffold: source 2dspgemm
