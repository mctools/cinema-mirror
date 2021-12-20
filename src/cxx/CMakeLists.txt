################################################################################
##                                                                            ##
##  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        ##
##                                                                            ##
##  Copyright 2021-2022 Prompt developers                                     ##
##                                                                            ##
##  Licensed under the Apache License, Version 2.0 (the "License");           ##
##  you may not use this file except in compliance with the License.          ##
##  You may obtain a copy of the License at                                   ##
##                                                                            ##
##      http://www.apache.org/licenses/LICENSE-2.0                            ##
##                                                                            ##
##  Unless required by applicable law or agreed to in writing, software       ##
##  distributed under the License is distributed on an "AS IS" BASIS,         ##
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  ##
##  See the License for the specific language governing permissions and       ##
##  limitations under the License.                                            ##
##                                                                            ##
################################################################################

project(prompt_core)


#add source
file(GLOB COR_SRC "${PROJECT_SOURCE_DIR}/Core/libsrc/*.cc")
file(GLOB RND_SRC "${PROJECT_SOURCE_DIR}/Rand/libsrc/*.cc")
file(GLOB UTL_SRC "${PROJECT_SOURCE_DIR}/Utils/libsrc/*.cc")
file(GLOB ANA_SRC "${PROJECT_SOURCE_DIR}/Ana/libsrc/*.cc")
file(GLOB PAR_SRC "${PROJECT_SOURCE_DIR}/Particles/libsrc/*.cc")
file(GLOB MOD_SRC "${PROJECT_SOURCE_DIR}/Models/libsrc/*.cc")
file(GLOB GEO_SRC "${PROJECT_SOURCE_DIR}/Geometry/libsrc/*.cc")
file(GLOB GUN_SRC "${PROJECT_SOURCE_DIR}/Gun/libsrc/*.cc")
file(GLOB MAT_SRC "${PROJECT_SOURCE_DIR}/Material/libsrc/*.cc")
file(GLOB PYN_SRC "${PROJECT_SOURCE_DIR}/Python/libsrc/*.cc")

ADD_LIBRARY(${PROJECT_NAME} SHARED ${COR_SRC} ${RND_SRC} ${UTL_SRC} ${ANA_SRC} ${PAR_SRC} ${MOD_SRC} ${GEO_SRC} ${MAT_SRC} ${PYN_SRC} ${GUN_SRC})


target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads NCrystal::NCrystal VecGeom::vecgeom VecGeom::vgdml)

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
    ${PROJECT_SOURCE_DIR}/Gun/libinc
    ${PROJECT_SOURCE_DIR}/Python/libinc
    ${VECGEOM_INCLUDE_DIRS}
    ${NCrystal_INCDIR}
)
