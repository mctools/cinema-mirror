#!/bin/bash
set -e
set -x

# export LD="x86_64-apple-darwin13.4.0-ld"
# CMAKE_ARGS="-DCMAKE_LINKER=$BUILD_PREFIX/bin/x86_64-apple-darwin13.4.0-ld"
GITREPOPREFIX="https://gitlab.com/cinema-developers"
NUMCPU=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')
if [ "${target_platform}" != "${build_platform}" ]
then
    CMAKE_PLATFORM_FLAGS="-DCMAKE_TOOLCHAIN_FILE=${RECIPE_DIR}/cross-linux.cmake"
else
    CMAKE_PLATFORM_FLAGS=""
fi

# mkdir ${SRC_DIR}/external
cd ${SRC_DIR}/external


git clone ${GITREPOPREFIX}/ncrystal.git
cd -
mkdir ${SRC_DIR}/external/ncrystal/build && cd ${SRC_DIR}/external/ncrystal/build
cmake  -DCMAKE_INSTALL_PREFIX=${PREFIX} ${CMAKE_PLATFORM_FLAGS} ${CMAKE_ARGS} ..
make -j ${NUMCPU} && make install
cd -
.  ${PREFIX}/setup.sh
export NCRYSTAL_DATA_PATH="${SP_DIR}/Cinema/ncmat:${SRC_DIR}/share/Ncrystal/data"

cd ${SRC_DIR}/external
git clone ${GITREPOPREFIX}/mcpl.git
mkdir ${SRC_DIR}/external/mcpl/build && cd ${SRC_DIR}/external/mcpl/build
cmake  -DCMAKE_INSTALL_PREFIX=${PREFIX} ${CMAKE_PLATFORM_FLAGS} ${CMAKE_ARGS} ..
make -j ${NUMCPU} && make install
cd -

git clone -b v3.2.3 --single-branch ${GITREPOPREFIX}/xerces-c.git
mkdir ${SRC_DIR}/external/xerces-c/build && cd ${SRC_DIR}/external/xerces-c/build
cmake -DBUILD_SHARED_LIBS=ON -Dnetwork=OFF -Dextra-warnings=OFF  -DCMAKE_INSTALL_PREFIX=${PREFIX} ${CMAKE_PLATFORM_FLAGS} ${CMAKE_ARGS} ..
make -j ${NUMCPU} && make install
cd -

git clone -b v1.2.0 --single-branch ${GITREPOPREFIX}/VecGeom.git
mkdir ${SRC_DIR}/external/VecGeom/build && cd ${SRC_DIR}/external/VecGeom/build
patch ${SRC_DIR}/external/VecGeom/persistency/gdml/source/src/Middleware.cpp < ${SRC_DIR}/external/vecgoem1.2.0_Middleware_cpp.patch
patch ${SRC_DIR}/external/VecGeom/persistency/gdml/source/include/Middleware.h < ${SRC_DIR}/external/vecgeom1.2.0_Middleware_h.patch
cmake -DXercesC_INCLUDE_DIR=${PREFIX}/include  ${CMAKE_PLATFORM_FLAGS} -DVECGEOM_BUILTIN_VECCORE=ON -DVECGEOM_FAST_MATH=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${PREFIX} -DVECGEOM_GDML=ON -DVECGEOM_USE_NAVINDEX=ON ${CMAKE_ARGS}  ..
make -j ${NUMCPU} && make install
cd -

cd ${SRC_DIR}
python -m pip install . --no-deps --ignore-installed --no-cache-dir -vvv 
cmake -DHDF5_DIR=${PREFIX} \
    -DCMAKE_INSTALL_PREFIX=${SP_DIR}/Cinema \
    -DCMAKE_PREFIX_PATH=${PREFIX} \
    ${CMAKE_PLATFORM_FLAGS} \
    ${SRC_DIR}
make -j ${NUMCPU} && make install
cd ${SRC_DIR}

# https://conda.io/projects/conda-build/en/latest/resources/compiler-tools.html#an-aside-on-cmake-and-sysroots
# CMAKE_PLATFORM_FLAGS+=(-DCMAKE_TOOLCHAIN_FILE="${RECIPE_DIR}/cross-linux.cmake")

# cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} \
#   ${CMAKE_PLATFORM_FLAGS} \
#   ${SRC_DIR}