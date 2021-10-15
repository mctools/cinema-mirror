project(prompt_core)


#add source
file(GLOB COR_SRC "${PROJECT_SOURCE_DIR}/Core/libsrc/*.cc")
file(GLOB RND_SRC "${PROJECT_SOURCE_DIR}/Rand/libsrc/*.cc")
file(GLOB UTL_SRC "${PROJECT_SOURCE_DIR}/Utils/libsrc/*.cc")
file(GLOB ANA_SRC "${PROJECT_SOURCE_DIR}/Ana/libsrc/*.cc")


ADD_LIBRARY(${PROJECT_NAME} SHARED ${COR_SRC} ${RND_SRC} ${UTL_SRC} ${ANA_SRC})


target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads VecGeom::vecgeom VecGeom::vgdml OpenMP::OpenMP_CXX)

target_include_directories( ${PROJECT_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/Core/libinc
    ${PROJECT_SOURCE_DIR}/Rand/libinc
    ${PROJECT_SOURCE_DIR}/Utils/libinc
    ${PROJECT_SOURCE_DIR}/Ana/libinc
    ${VECGEOM_INCLUDE_DIRS}
)
