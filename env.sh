#!/bin/bash

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
  else
    .  $CINEMAPATH/external/ncrystal/install/setup.sh
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
    git clone https://gitlab.com/xxcai1/VecGeom.git
    cd -
    mkdir $CINEMAPATH/external/VecGeom/build && cd $CINEMAPATH/external/VecGeom/build
    cmake  -DCMAKE_INSTALL_PREFIX=$CINEMAPATH/external/VecGeom/install -DGDML=On -DUSE_NAVINDEX=On  ..
    make -j${NUMCPU} && make install
    cd -
    echo "installed  VecGeom"
  else
    echo "Found VecGeom"
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
    pip install -r $CINEMAPATH/requirement
  fi
fi

#SSSP
if [ -f $CINEMAPATH/external/pixiusssp/sssp_efficiency.json ]; then
  export PIXIUSSSP=$CINEMAPATH/external/pixiusssp
else
  read -r -p "Do you want install the SSSP DFT pseudopotential into $CINEMAPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    cd  $CINEMAPATH/external/
    git clone https://gitlab.com/xxcai1/pixiusssp.git
    cd -
    cd $CINEMAPATH/external/pixiusssp
    tar -xzvf SSSP_efficiency_pseudos.tar.gz
    mkdir SSSP_precision_pseudos
    tar -xzvf SSSP_precision_pseudos.tar.gz --directory SSSP_precision_pseudos
    cd -
    export PIXIUSSSP=$CINEMAPATH/external/pixiusssp
  fi
fi

# ln -s
echo -e "Enjoy!"
