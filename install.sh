#!/bin/bash

# --set options for git repos
while getopts ":f" option
  do
    case "${option}"
     in
      f) echo "Running: "${PREFIX};;
      *) PREFIX="https://gitlab.com/cinema-developers";;
    esac
done
if (( $OPTIND == 1 )); then
  PREFIX="https://gitlab.com/cinema-developers"
fi 
NUMCPU=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')

#remove previously sourced env
function cinema_prunepath() {
    P=$(IFS=:;for p in ${!1}; do [[ $p != ${2}* ]] && echo -n ":$p" || :; done)
    export $1=${P:1:99999}
}

if [ ! -z ${CINEMAPATH:-} ]; then
    cinema_prunepath PATH "$CINEMAPATH/cinemabin"
    cinema_prunepath LD_LIBRARY_PATH "$CINEMAPATH/cinemabin:$CINEMAPATH/src/python/bin"
    cinema_prunepath DYLD_LIBRARY_PATH "$CINEMAPATH/cinemabin"
    cinema_prunepath PYTHONPATH "$CINEMAPATH/src/python;$CINEMAPATH/src/python/ptgeo/python"
    echo "Cleaned up previously defined Cinema enviorment"
fi

unset cinema_prunepath

export CINEMAPATH="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#install ncrystal
export response='y'
if [ ! -f $CINEMAPATH/external/ncrystal/install/lib/libNCrystal.so ]; then
  # read -r -p "Do you want to install NCrystal into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      if [ ! -d $CINEMAPATH/external ]; then
        mkdir $CINEMAPATH/external
      fi
      cd $CINEMAPATH/external
      if [ -d ncrystal ]; then
        rm -rf ncrystal
      fi
      git clone ${PREFIX}/ncrystal.git
      cd -
      mkdir $CINEMAPATH/external/ncrystal/build && cd $CINEMAPATH/external/ncrystal/build
      cmake  -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/ncrystal/install ..
      make -j ${NUMCPU} && make install
      cd -
      echo "installed  ncrystal"
    else
      echo "Found ncrystal"
    fi
    .  $CINEMAPATH/external/ncrystal/install/setup.sh
    export NCRYSTAL_DATA_PATH="$CINEMAPATH/ncmat:$CINEMAPATH/external/ncystal/install/share/Ncrystal/data"
  else
    .  $CINEMAPATH/external/ncrystal/install/setup.sh
    export NCRYSTAL_DATA_PATH="$CINEMAPATH/ncmat:$CINEMAPATH/external/ncystal/install/share/Ncrystal/data"
  fi

#MCPL
if [ ! -f $CINEMAPATH/external/KDSource/install/lib/libmcpl.so ]; then
  # read -r -p "Do you want to install MCPL into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      if [ ! -d $CINEMAPATH/external ]; then
        mkdir $CINEMAPATH/external
      fi
      cd $CINEMAPATH/external
      if [ -d mcpl ]; then
        rm -rf mcpl
      fi
      git clone ${PREFIX}/mcpl.git
      cd -
      # MCPL is not built as it will be built in the KDSource
      # mkdir $CINEMAPATH/external/mcpl/build && cd $CINEMAPATH/external/mcpl/build
      # cmake  -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/mcpl/install ..
      # make -j ${NUMCPU} && make install
      # cd -
      echo "installed  MCPL"
    else
      echo "Found MCPL"
    fi
  fi

#install libxerces
if [ ! -f $CINEMAPATH/external/xerces-c/install/lib/libxerces-c.so ]; then
  # read -r -p "Do you want to install libxerces into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      if [ ! -d $CINEMAPATH/external ]; then
        mkdir $CINEMAPATH/external
      fi
      cd $CINEMAPATH/external
      if [ -d xerces-c ]; then
        rm -rf xerces-c
      fi
      git clone -b v3.2.3 --single-branch ${PREFIX}/xerces-c.git
      cd -
      mkdir $CINEMAPATH/external/xerces-c/build && cd $CINEMAPATH/external/xerces-c/build
      cmake  -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/ncrystal/install ..
      cmake -DBUILD_SHARED_LIBS=ON -Dnetwork=OFF -Dextra-warnings=OFF  -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/xerces-c/install ..
      make -j ${NUMCPU} && make install
      cd -
      echo "installed  libxerces"
    fi
fi

# KDSource 
if [ ! -f $CINEMAPATH/external/KDSource/install/lib/libkdsource.so ]; then
  # read -r -p "Do you want to install KDSource into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      if [ ! -d $CINEMAPATH/external ]; then
        mkdir $CINEMAPATH/external
      fi
      cd $CINEMAPATH/external
      if [ -d KDSource ]; then
        rm -rf KDSource
      fi
      git clone https://gitlab.com/cinema-developers/KDSource.git
      cd -
      mkdir $CINEMAPATH/external/KDSource/build $CINEMAPATH/external/KDSource/install && cd $CINEMAPATH/external/KDSource/build
      rm -rf $CINEMAPATH/external/KDSource/mcpl
      ln -s $CINEMAPATH/external/mcpl $CINEMAPATH/external/KDSource/
      cmake  -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/KDSource/install ..
      make -j ${NUMCPU} && make install
      cd -
      echo "installed  KDSource"
    fi
fi

# gidipuls
if [ ! -f $CINEMAPATH/external/gidiplus/lib/libgidiplus.a ]; then
  read -r -p "Do you want to install gidiplus into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    if [ ! -d $CINEMAPATH/external ]; then
      mkdir $CINEMAPATH/external
    fi
    cd $CINEMAPATH/external
    if [ -d gidiplus ]; then
      rm -rf gidiplus
    fi
    git clone -b pt --single-branch ${PREFIX}/gidiplus.git
    cd -
    cd $CINEMAPATH/external/gidiplus
    wget https://github.com/zeux/pugixml/releases/download/v1.13/pugixml-1.13.zip 
    mv pugixml-1.13.zip Misc
    make -s -j${NUMCPU} CXXFLAGS="-std=c++11 -fPIC"  CFLAGS="-fPIC" HDF5_LIB=/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so HDF5_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial/
    cd -
    echo "installed gidiplus"
  fi
fi



#install VecGeom

if [ ! -f $CINEMAPATH/external/VecGeom/install/lib/libvecgeom.a ]; then
  # read -r -p "Do you want to install VecGeom into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    if [ ! -d $CINEMAPATH/external ]; then
      mkdir $CINEMAPATH/external
    fi
    cd $CINEMAPATH/external
    if [ -d VecGeom ]; then
      rm -rf VecGeom
    fi
    git clone -b v1.2.6 --single-branch ${PREFIX}/VecGeom.git
    cd -
    mkdir $CINEMAPATH/external/VecGeom/build && cd $CINEMAPATH/external/VecGeom/build
    # patch $CINEMAPATH/external/VecGeom/persistency/gdml/source/src/Middleware.cpp < $CINEMAPATH/external/vecgoem1.2.0_Middleware_cpp.patch
    # patch $CINEMAPATH/external/VecGeom/persistency/gdml/source/include/Middleware.h < $CINEMAPATH/external/vecgeom1.2.0_Middleware_h.patch
    cmake -DXercesC_INCLUDE_DIR=$CINEMAPATH/external/xerces-c/install/include   -DVECGEOM_BUILTIN_VECCORE=ON -DVECGEOM_FAST_MATH=OFF -DBUILD_TESTING=OFF -DXercesC_LIBRARY_RELEASE=$CINEMAPATH/external/xerces-c/install/lib/libxerces-c.so -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/VecGeom/install -DVECGEOM_GDML=ON -DVECGEOM_USE_NAVINDEX=ON  ..
    make -j${NUMCPU} && make install
    cd -
    echo "installed  VecGeom"
  fi
fi

if [ -f $CINEMAPATH/src/python/Cinema/__init__.py ]; then
  export PYTHONPATH="$CINEMAPATH/src/python:$CINEMAPATH/src/python/ptgeo/python:$PYTHONPATH"
  echo "Added Cinema python module into path"
else
  echo "Can not find Cinema python module"
fi


if [ ! -d $CINEMAPATH/cinemabin ]; then
  mkdir $CINEMAPATH/cinemabin
  cd $CINEMAPATH/cinemabin
  cmake .. 
  make -j${NUMCPU}
  cd -
fi


if [ -d $CINEMAPATH/cinemabin ]; then
  export LD_LIBRARY_PATH="$CINEMAPATH/cinemabin:$LD_LIBRARY_PATH"
  export DYLD_LIBRARY_PATH="$CINEMAPATH/cinemabin:$DYLD_LIBRARY_PATH"
  export PATH="$CINEMAPATH/src/python/ptgeo/examples:$CINEMAPATH/cinemabin/bin:$CINEMAPATH/scripts:$PATH"
  echo "Added the cinemabin directory into environment"
fi


if [ -f $CINEMAPATH/cinemavirenv/bin/activate ]; then
  . $CINEMAPATH/cinemavirenv/bin/activate
else
  # read -r -p "Do you want install Cinema python virtual environment in $CINEMAPATH/cinemavirenv? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    python3 -m venv $CINEMAPATH/cinemavirenv
    . $CINEMAPATH/cinemavirenv/bin/activate
    if [[ $DOCKER == false ]]; then
      pip install -r $CINEMAPATH/requirement
    else
      echo "Python with Docker enviorment"
      pip install -r $CINEMAPATH/requirement
    fi
  fi
fi

#SSSP
if [ -f $CINEMAPATH/external/pixiusssp/SSSP_precision_pseudos/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF ]; then
  export PIXIUSSSP=$CINEMAPATH/external/pixiusssp
else
  # read -r -p "Do you want (re)install the SSSP DFT pseudopotential into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    cd  $CINEMAPATH/external/
    git clone ${PREFIX}/pixiusssp.git
    cd -
    cd $CINEMAPATH/external/pixiusssp
    mkdir SSSP_precision_pseudos
    tar -xzvf SSSP_1.1.2_PBE_precision.tar.gz --directory SSSP_precision_pseudos
    cp Cu.pbe-dn-kjpaw_psl.1.0.0.UPF SSSP_precision_pseudos/
    rm SSSP_precision_pseudos/Cu_ONCV_PBE-1.0.oncvpsp.upf
    cd -
    export PIXIUSSSP=$CINEMAPATH/external/pixiusssp
  fi
fi

# add submodule and define the master branch as the one you want to track
if [ ! -f $CINEMAPATH/src/python/ptgeo/__init__.py ]; then
  echo "Clone ptgeo submodule"
  cd $CINEMAPATH/src/python
  # git submodule add https://gitlab.com/cinema-developers/ptgeo.git
  # git submodule init
  git submodule update --init --recursive
  cd -
fi


export PATH="$CINEMAPATH/src/python/ptgeo/examples:$PATH"
echo "Added the ptgeo example directory into environment"
