
cmake_minimum_required( VERSION 3.0 )

project( libstatespace )
find_package( Eigen3 )
include_directories( /usr/local/include /usr/include/mkl ${EIGEN3_INCLUDE_DIRS} )
link_directories( /usr/local/lib /usr/lib )

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_FLAGS "-mavx -mfma -openmp -DEIGEN_USE_MKL_ALL" )
set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Ofast" )
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast" )

add_subdirectory( pybind11 )

pybind11_add_module( libstatespace SHARED api.cpp )

target_link_libraries( libstatespace PRIVATE blas lapack )

