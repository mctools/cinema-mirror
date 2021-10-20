project(prompt_core)


#add source
file(GLOB COR_SRC "${PROJECT_SOURCE_DIR}/Core/libsrc/*.cc")
file(GLOB RND_SRC "${PROJECT_SOURCE_DIR}/Rand/libsrc/*.cc")
file(GLOB UTL_SRC "${PROJECT_SOURCE_DIR}/Utils/libsrc/*.cc")
file(GLOB ANA_SRC "${PROJECT_SOURCE_DIR}/Ana/libsrc/*.cc")
file(GLOB PAR_SRC "${PROJECT_SOURCE_DIR}/Particles/libsrc/*.cc")
file(GLOB MOD_SRC "${PROJECT_SOURCE_DIR}/Models/libsrc/*.cc")
file(GLOB GEO_SRC "${PROJECT_SOURCE_DIR}/Geometry/libsrc/*.cc")
file(GLOB MAT_SRC "${PROJECT_SOURCE_DIR}/Material/libsrc/*.cc")

ADD_LIBRARY(${PROJECT_NAME} SHARED ${COR_SRC} ${RND_SRC} ${UTL_SRC} ${ANA_SRC} ${PAR_SRC} ${MOD_SRC} ${GEO_SRC} ${MAT_SRC})


target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads NCrystal::NCrystal VecGeom::vecgeom VecGeom::vgdml OpenMP::OpenMP_CXX)

target_include_directories( ${PROJECT_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/Core/libinc
    ${PROJECT_SOURCE_DIR}/Rand/libinc
    ${PROJECT_SOURCE_DIR}/Utils/libinc
    ${PROJECT_SOURCE_DIR}/Ana/libinc
    ${PROJECT_SOURCE_DIR}/Particles/libinc
    ${PROJECT_SOURCE_DIR}/Models/libinc
    ${PROJECT_SOURCE_DIR}/Geometry/libinc
    ${PROJECT_SOURCE_DIR}/Material/libinc
    ${VECGEOM_INCLUDE_DIRS}
    ${NCrystal_INCDIR}
)
