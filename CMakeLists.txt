cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(prompt)

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
