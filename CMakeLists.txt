cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cinema)

find_package(HDF5 REQUIRED COMPONENTS C)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

#Concurrent
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# VecGeom
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/VecGeom/install/")
set(VecGeom_VERSION 1.1.17)
find_package(VecGeom ${VecGeom_VERSION} REQUIRED)
message(STATUS "Using VecGeom version ${VecGeom_VERSION}")
# make sure we import VecGeom architecture flags - is this needed?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECGEOM_CXX_FLAGS}")
# Make sure VecGeom::vgdml is enabled
if(NOT TARGET VecGeom::vgdml)
  message(FATAL_ERROR "Prompt requires VecGeom compiled with GDML support")
endif()

# NCrystal
set(NCrystal_DIR ${CMAKE_SOURCE_DIR}/external/ncrystal/install/lib/cmake)
find_package(NCrystal 3.0.0 REQUIRED)



if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(src/cxx)
add_subdirectory(bin)

#Test
enable_testing()
add_subdirectory(src/cxxtests)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

file(GLOB TESTFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pythontests/*.py")
foreach(TS ${TESTFILES})
	get_filename_component(EXCNAME  ${TS}  NAME_WE)
	add_test (NAME python-${EXCNAME}
	  COMMAND ${Python_EXECUTABLE} ${TS}
	  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cinemabin #fixme find a temp dir defined in the enviroment
	)
endforeach(TS ${TESTFILES})
