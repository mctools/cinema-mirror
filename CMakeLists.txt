cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(cinema)
set(FIND_LIBRARY_USE_LIB64_PATHS TRUE)
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
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/ncrystal/install/")
message(STATUS "NCrystal_DIR=$ENV{NCrystal_ext}/lib/cmake")
set(NCrystal_DIR  "$ENV{NCrystal_ext}/lib/cmake")
find_package(NCrystal 3.8.0 REQUIRED)
message(STATUS "FOUND NCrystal: ${NCrystal_FOUND} ${NCrystal_DIR}")

# MCPL
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/KDSource/install")
set(MCPL_DIR  "${CMAKE_SOURCE_DIR}/external/KDSource/install/lib/cmake")
find_package(MCPL REQUIRED)
message(STATUS "FOUND MCPL: ${MCPL_FOUND} ${MCPL_DIR}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# KDSrouce 
set(KDS_INCDIR "${PROJECT_SOURCE_DIR}/external/KDSource/install/include") 
set(KDS_LIB "${PROJECT_SOURCE_DIR}/external/KDSource/install/lib") 
# set(KDS_SRC "${PROJECT_SOURCE_DIR}/external/KDSource/src/kdsource") 

# GIDIPLUS
set(GIDIPLUS_INCDIR "${PROJECT_SOURCE_DIR}/external/gidiplus/include") 
set(GIDIPLUS_LIB "${PROJECT_SOURCE_DIR}/external/gidiplus/lib") 

# set(Python_ROOT_DIR ${CMAKE_SOURCE_DIR}/cinemavirenv/bin/python)
set(Python_FIND_STRATEGY LOCATION)

#Test
enable_testing()

add_subdirectory(src/cxx)
add_subdirectory(src/cxxtests) #DO NOT REMOVE TESTS!!!
add_subdirectory(bin)


set(Python_FIND_STRATEGY LOCATION)
set(Python_ROOT_DIR ${PROJECT_SOURCE_DIR}/cinemavirenv/bin/python)
find_package(Python REQUIRED COMPONENTS Interpreter)

file(GLOB_RECURSE TESTFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pythontests/test_*.py")
foreach(TS ${TESTFILES})
	message(STATUS "Running ${Python_EXECUTABLE} ${TS}")
	get_filename_component(EXCNAME  ${TS}  NAME_WE)
	add_test (NAME python-${EXCNAME}
	  COMMAND ${Python_EXECUTABLE} ${TS}
	  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cinemabin #fixme find a temp dir defined in the enviroment
	)
endforeach(TS ${TESTFILES})