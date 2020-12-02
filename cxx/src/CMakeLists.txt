project(PiXiu)

list(APPEND EXTRA_INCLUDE_DIR "")
add_compile_options ("-fopenmp")

#OpenMP
find_package(OpenMP REQUIRED)


#add source

file(GLOB HIS_SRC "${PROJECT_SOURCE_DIR}/Hist/libsrc/*.cc")


ADD_LIBRARY(${PROJECT_NAME} SHARED ${HIS_SRC})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads OpenMP::OpenMP_CXX)


target_include_directories( ${PROJECT_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/Hist/libinc
)
