#!/bin/bash

DOCKER=false
while getopts hi:x:d option
do
case "${option}"
in
h) echo "$USAGE"
   exit 0;;
x) INPUT=${OPTARG};;
i) ARGS=${OPTARG};;
d) DOCKER=true;;
esac
done


if ! hash cmake 2>/dev/null
then
    echo "cmake is required to continue the installation of Cinema"
    return
fi

if ! hash git 2>/dev/null
then
    echo "git is required to continue the installation of Cinema"
    return
fi

if ! hash pw.x 2>/dev/null
then
    echo "pw.x is required to continue the installation of Cinema"
    return
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
    cinema_prunepath PYTHONPATH "$CINEMAPATH/src/python"
    echo "Cleaned up previously defined Cinema enviorment"
fi

unset cinema_prunepath

export CINEMAPATH="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#install ncrystal

if [ ! -f $CINEMAPATH/external/ncrystal/install/lib/libNCrystal.so ]; then
  read -r -p "Do you want to install NCrystal into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      if [ ! -d $CINEMAPATH/external ]; then
        mkdir $CINEMAPATH/external
      fi
      cd $CINEMAPATH/external
      if [ -d ncrystal ]; then
        rm -rf ncrystal
      fi
      git clone https://gitlab.com/xxcai1/ncrystal.git
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
if [ ! -f $CINEMAPATH/external/mcpl/install/lib/libmcpl.so ]; then
  read -r -p "Do you want to install MCPL into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      if [ ! -d $CINEMAPATH/external ]; then
        mkdir $CINEMAPATH/external
      fi
      cd $CINEMAPATH/external
      if [ -d mcpl ]; then
        rm -rf mcpl
      fi
      git clone https://gitlab.com/cinema-developers/mcpl
      cd -
      mkdir $CINEMAPATH/external/mcpl/build && cd $CINEMAPATH/external/mcpl/build
      cmake  -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/mcpl/install ..
      make -j ${NUMCPU} && make install
      cd -
      echo "installed  MCPL"
    else
      echo "Found MCPL"
    fi
  fi

#install libxerces
if [ ! -f $CINEMAPATH/external/xerces-c/install/lib/libxerces-c.so ]; then
  read -r -p "Do you want to install libxerces into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      if [ ! -d $CINEMAPATH/external ]; then
        mkdir $CINEMAPATH/external
      fi
      cd $CINEMAPATH/external
      if [ -d xerces-c ]; then
        rm -rf xerces-c
      fi
      git clone -b v3.2.3 --single-branch https://gitlab.com/xxcai1/xerces-c.git
      cd -
      mkdir $CINEMAPATH/external/xerces-c/build && cd $CINEMAPATH/external/xerces-c/build
      cmake  -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/ncrystal/install ..
      cmake -DBUILD_SHARED_LIBS=ON -Dnetwork=OFF -Dextra-warnings=OFF  -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/xerces-c/install ..
      make -j ${NUMCPU} && make install
      cd -
      echo "installed  libxerces"
    fi
fi


#install VecGeom

if [ ! -f $CINEMAPATH/external/VecGeom/install/lib/libvecgeom.a ]; then
  read -r -p "Do you want to install VecGeom into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    if [ ! -d $CINEMAPATH/external ]; then
      mkdir $CINEMAPATH/external
    fi
    cd $CINEMAPATH/external
    if [ -d VecGeom ]; then
      rm -rf VecGeom
    fi
    git clone -b v1.2.0 --single-branch https://gitlab.com/xxcai1/VecGeom.git
    cd -
    mkdir $CINEMAPATH/external/VecGeom/build && cd $CINEMAPATH/external/VecGeom/build
    patch $CINEMAPATH/external/VecGeom/persistency/gdml/source/src/Middleware.cpp < $CINEMAPATH/external/vecgoem1.2.0_Middleware_cpp.patch
    patch $CINEMAPATH/external/VecGeom/persistency/gdml/source/include/Middleware.h < $CINEMAPATH/external/vecgeom1.2.0_Middleware_h.patch
    cmake -DXercesC_INCLUDE_DIR=$CINEMAPATH/external/xerces-c/install/include   -DVECGEOM_BUILTIN_VECCORE=ON -DVECGEOM_FAST_MATH=OFF -DBUILD_TESTING=OFF -DXercesC_LIBRARY_RELEASE=$CINEMAPATH/external/xerces-c/install/lib/libxerces-c.so -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/VecGeom/install -DVECGEOM_GDML=ON -DVECGEOM_USE_NAVINDEX=ON  ..
    make -j${NUMCPU} && make install
    cd -
    echo "installed  VecGeom"
  fi
fi

if [ -f $CINEMAPATH/src/python/Cinema/__init__.py ]; then
  export PYTHONPATH="$CINEMAPATH/src/python:$PYTHONPATH"
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
  #export PATH="$CINEMAPATH/resource/cxx/src:$PATH"
  #export PATH="$CINEMAPATH/resource/src/bin:$PATH"
  export LD_LIBRARY_PATH="$CINEMAPATH/cinemabin:$LD_LIBRARY_PATH"
  export DYLD_LIBRARY_PATH="$CINEMAPATH/cinemabin:$DYLD_LIBRARY_PATH"
  export PATH="$CINEMAPATH/cinemabin/bin:$CINEMAPATH/scripts:$PATH"
  echo "Added the cinemabin directory into environment"
fi


if [ -f $CINEMAPATH/cinemavirenv/bin/activate ]; then
  . $CINEMAPATH/cinemavirenv/bin/activate
else
  read -r -p "Do you want install Cinema python virtual environment in $CINEMAPATH/cinemavirenv? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    python3 -m venv $CINEMAPATH/cinemavirenv
    . $CINEMAPATH/cinemavirenv/bin/activate
    if [[ $DOCKER == false ]]; then
      pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r $CINEMAPATH/requirement
    else
      echo "Python with Docker enviorment"
      pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r $CINEMAPATH/requirement.docker
    fi
  fi
fi

#SSSP
if [ -f $CINEMAPATH/external/pixiusssp/SSSP_precision_pseudos/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF ]; then
  export PIXIUSSSP=$CINEMAPATH/external/pixiusssp
else
  read -r -p "Do you want (re)install the SSSP DFT pseudopotential into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    cd  $CINEMAPATH/external/
    git clone https://gitlab.com/xxcai1/pixiusssp.git
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

# ln -s
echo -e "Enjoy!"
