#!/bin/bash

GITREPOPREFIX="https://gitlab.com/cinema-developers"
NUMCPU=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')

mkdir ${SRC_DIR}/external
cd ${SRC_DIR}/external

git clone ${GITREPOPREFIX}/ncrystal.git
cd -
mkdir ${SRC_DIR}/external/ncrystal/build && cd ${SRC_DIR}/external/ncrystal/build
cmake  -DCMAKE_INSTALL_PREFIX=${SRC_DIR}/external/ncrystal/install ..
make -j ${NUMCPU} && make install
cd -
.  ${SRC_DIR}/external/ncrystal/install/setup.sh
export NCRYSTAL_DATA_PATH="${SRC_DIR}/ncmat:${SRC_DIR}/external/ncystal/install/share/Ncrystal/data"

git clone ${GITREPOPREFIX}/mcpl.git
mkdir ${SRC_DIR}/external/mcpl/build && cd ${SRC_DIR}/external/mcpl/build
cmake  -DCMAKE_INSTALL_PREFIX=${SRC_DIR}/external/mcpl/install ..
make -j ${NUMCPU} && make install
cd -

git clone -b v3.2.3 --single-branch ${GITREPOPREFIX}/xerces-c.git
mkdir ${SRC_DIR}/external/xerces-c/build && cd ${SRC_DIR}/external/xerces-c/build
cmake  -DCMAKE_INSTALL_PREFIX=${SRC_DIR}/external/ncrystal/install ..
cmake -DBUILD_SHARED_LIBS=ON -Dnetwork=OFF -Dextra-warnings=OFF  -DCMAKE_INSTALL_PREFIX=${SRC_DIR}/external/xerces-c/install ..
make -j ${NUMCPU} && make install
cd -

git clone -b v1.2.0 --single-branch ${GITREPOPREFIX}/VecGeom.git
mkdir ${SRC_DIR}/external/VecGeom/build && cd ${SRC_DIR}/external/VecGeom/build
patch ${SRC_DIR}/external/VecGeom/persistency/gdml/source/src/Middleware.cpp < ${SRC_DIR}/external/vecgoem1.2.0_Middleware_cpp.patch
patch ${SRC_DIR}/external/VecGeom/persistency/gdml/source/include/Middleware.h < ${SRC_DIR}/external/vecgeom1.2.0_Middleware_h.patch
cmake -DXercesC_INCLUDE_DIR=${SRC_DIR}/external/xerces-c/install/include   -DVECGEOM_BUILTIN_VECCORE=ON -DVECGEOM_FAST_MATH=OFF -DBUILD_TESTING=OFF -DXercesC_LIBRARY_RELEASE=${SRC_DIR}/external/xerces-c/install/lib/libxerces-c.so -DCMAKE_INSTALL_PREFIX=${SRC_DIR}/external/VecGeom/install -DVECGEOM_GDML=ON -DVECGEOM_USE_NAVINDEX=ON  ..
make -j ${NUMCPU} && make install
cd -

cd ${SRC_DIR}
cmake .
make -j ${NUMCPU}