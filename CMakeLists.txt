cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(prompt)
find_package(HDF5 REQUIRED COMPONENTS C)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(src/cxx)

#Test
enable_testing()
add_subdirectory(src/cxxtest)

#find_package(Python REQUIRED COMPONENTS Interpreter Development)

file(GLOB TESTFILES "${PROJECT_SOURCE_DIR}/bin/*.cc" "${PROJECT_SOURCE_DIR}/tests/*.c")
foreach(TS ${TESTFILES})
  get_filename_component(EXCNAME  ${TS}  NAME_WE)
  add_executable( ${EXCNAME} ${TS} )
	#link_directories(/usr/lib/x86_64-linux-gnu)
  #target_link_libraries( ${EXCNAME} ${PROJECT_NAME} gsl )
endforeach(TS ${TESTFILES})
