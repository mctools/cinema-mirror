#!/bin/bash

GITREPOPREFIX="https://gitlab.com/cinema-developers"

mkdir $PREFIX/external
cd $PREFIX/external

git clone ${GITREPOPREFIX}/ncrystal.git
cd -
mkdir $PREFIX/external/ncrystal/build && cd $PREFIX/external/ncrystal/build
cmake  -DCMAKE_INSTALL_PREFIX=$PREFIX/external/ncrystal/install ..
make && make install
cd -
.  $PREFIX/external/ncrystal/install/setup.sh
export NCRYSTAL_DATA_PATH="$PREFIX/ncmat:$PREFIX/external/ncystal/install/share/Ncrystal/data"

git clone ${GITREPOPREFIX}/mcpl.git
cd -
mkdir $PREFIX/external/mcpl/build && cd $PREFIX/external/mcpl/build
cmake  -DCMAKE_INSTALL_PREFIX=$PREFIX/external/mcpl/install ..
make && make install
cd -

git clone -b v3.2.3 --single-branch ${GITREPOPREFIX}/xerces-c.git
cd -
mkdir $PREFIX/external/xerces-c/build && cd $PREFIX/external/xerces-c/build
cmake  -DCMAKE_INSTALL_PREFIX=$PREFIX/external/ncrystal/install ..
cmake -DBUILD_SHARED_LIBS=ON -Dnetwork=OFF -Dextra-warnings=OFF  -DCMAKE_INSTALL_PREFIX=$PREFIX/external/xerces-c/install ..
make && make install
cd -

git clone -b v1.2.0 --single-branch ${GITREPOPREFIX}/VecGeom.git
cd -
mkdir $PREFIX/external/VecGeom/build && cd $PREFIX/external/VecGeom/build
patch $PREFIX/external/VecGeom/persistency/gdml/source/src/Middleware.cpp < $PREFIX/external/vecgoem1.2.0_Middleware_cpp.patch
patch $PREFIX/external/VecGeom/persistency/gdml/source/include/Middleware.h < $PREFIX/external/vecgeom1.2.0_Middleware_h.patch
cmake -DXercesC_INCLUDE_DIR=$PREFIX/external/xerces-c/install/include   -DVECGEOM_BUILTIN_VECCORE=ON -DVECGEOM_FAST_MATH=OFF -DBUILD_TESTING=OFF -DXercesC_LIBRARY_RELEASE=$PREFIX/external/xerces-c/install/lib/libxerces-c.so -DCMAKE_INSTALL_PREFIX=$PREFIX/external/VecGeom/install -DVECGEOM_GDML=ON -DVECGEOM_USE_NAVINDEX=ON  ..
make && make install
cd -

cd $PREFIX
cmake .
make