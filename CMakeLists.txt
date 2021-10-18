cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(prompt)

find_package(HDF5 REQUIRED COMPONENTS C)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

#OpenMP
find_package(OpenMP REQUIRED)
add_compile_options ("-fopenmp")

#Concurrent
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# VecGeom
set(CMAKE_PREFIX_PATH "/home/caixx/git/VecGeom/install") #fixme: use env instead
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
find_package(NCrystal 2.7.0 REQUIRED)


# Find Geant4, optional for now
find_package(Geant4 QUIET)
if(Geant4_FOUND)
  message(STATUS "Using Geant4 version ${Geant4_VERSION} from ${Geant4_INCLUDE_DIRS}")
else()
  message(STATUS "Did not find Geant4")
endif()


if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(src/cxx)
add_subdirectory(bin)

#Test
enable_testing()
add_subdirectory(src/cxxtest)

#find_package(Python REQUIRED COMPONENTS Interpreter Development)
