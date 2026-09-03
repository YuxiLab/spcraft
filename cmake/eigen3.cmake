function(find_eigen3)
    find_package(Eigen3 5.0.1 QUIET NO_MODULE)
    if(NOT Eigen3_FOUND)
        message(STATUS "Eigen3 not found, adding it via CPM")
        CPMAddPackage(
            NAME Eigen3
            URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
            VERSION 5.0.1
            OPTIONS
                "EIGEN_BUILD_DOC OFF"
                "EIGEN_BUILD_TESTS OFF"
        )
    else()
        message(STATUS "Eigen3 found, using existing installation")
    endif()

    # Export the Eigen3 target for use in subdirectories.
    if(TARGET Eigen3::Eigen)
        set(EIGEN3_TARGET Eigen3::Eigen PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Eigen3 not found")
    endif()
endfunction()
