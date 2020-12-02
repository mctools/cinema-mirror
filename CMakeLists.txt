cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(cxx/src)
add_subdirectory(cxx/tests)
