
cmake_minimum_required( VERSION 3.0 )

project( libstatespace )
find_package( Eigen3 )
include_directories( /usr/local/include /usr/include/mkl ${EIGEN3_INCLUDE_DIRS} )
link_directories( /usr/local/lib /usr/lib )

add_subdirectory( /tmp/pybind11 cmake-build-debug )
pybind11_add_module( libstatespace SHARED api.cpp )
