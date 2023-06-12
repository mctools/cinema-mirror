cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cinema)
find_package(HDF5 REQUIRED COMPONENTS C)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

#Concurrent
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# VecGeom
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/VecGeom/install/;${CMAKE_SOURCE_DIR}/external/xerces-c/install/")
set(VecGeom_VERSION 1.2.0)
find_package(VecGeom ${VecGeom_VERSION} REQUIRED)
message(STATUS "Using VecGeom version ${VecGeom_VERSION}")
# make sure we import VecGeom architecture flags - is this needed?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECGEOM_CXX_FLAGS}")
# Make sure VecGeom::vgdml is enabled
if(NOT TARGET VecGeom::vgdml)
  message(FATAL_ERROR "Prompt requires VecGeom compiled with GDML support")
endif()

# NCrystal
set(FIND_LIBRARY_USE_LIB64_PATHS TRUE)
message(STATUS "NCrystal_DIR=$ENV{NCrystal_ext}/lib/cmake")
set(NCrystal_DIR  "$ENV{NCrystal_ext}/lib/cmake")
find_package(NCrystal 3.0.0 REQUIRED)

# MCPL
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/mcpl/install/lib/cmake/MCPL/")
find_package(MCPL REQUIRED)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(src/cxx)
add_subdirectory(bin)

#Test
enable_testing()
add_subdirectory(src/cxxtests)


set(Python_ROOT_DIR ${PROJECT_SOURCE_DIR}/cinemavirenv/bin/python)
set(Python_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS Interpreter)

file(GLOB TESTFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pythontests/*.py")
foreach(TS ${TESTFILES})
	get_filename_component(EXCNAME  ${TS}  NAME_WE)
	add_test (NAME python-${EXCNAME}
	  COMMAND ${Python_EXECUTABLE} ${TS}
	  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cinemabin #fixme find a temp dir defined in the enviroment
	)
endforeach(TS ${TESTFILES})
