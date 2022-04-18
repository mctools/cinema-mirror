#!/bin/bash

if ! hash cmake 2>/dev/null
then
    echo "cmake is required to continue the installation of Prompt"
    return
fi

if ! hash git 2>/dev/null
then
    echo "git is required to continue the installation of Prompt"
    return
fi

NUMCPU=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')

#remove previously sourced env
function prompt_prunepath() {
    P=$(IFS=:;for p in ${!1}; do [[ $p != ${2}* ]] && echo -n ":$p" || :; done)
    export $1=${P:1:99999}
}

if [ ! -z ${PTPATH:-} ]; then
    prompt_prunepath PATH "$PTPATH/promptbin"
    prompt_prunepath LD_LIBRARY_PATH "$PTPATH/promptbin:$PTPATH/src/python/bin"
    prompt_prunepath DYLD_LIBRARY_PATH "$PTPATH/promptbin"
    prompt_prunepath PYTHONPATH "$PTPATH/src/python"
    echo "Cleaned up previously defined Prompt enviorment"
fi

unset prompt_prunepath

export PTPATH="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#install ncrystal

if [ ! -f $PTPATH/external/ncrystal/install/lib/libNCrystal.so ]; then
  read -r -p "Do you want to install NCrysta into $PTPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      if [ ! -d $PTPATH/external ]; then
        mkdir $PTPATH/external
      fi
      cd $PTPATH/external
      if [ -d ncrystal ]; then
        rm -rf ncrystal
      fi
      git clone https://gitlab.com/xxcai1/ncrystal.git
      cd -
      mkdir $PTPATH/external/ncrystal/build && cd $PTPATH/external/ncrystal/build
      cmake  -DCMAKE_INSTALL_PREFIX=$PTPATH/external/ncrystal/install ..
      make -j ${NUMCPU} && make install
      cd -
      echo "installed  ncrystal"
    else
      echo "Found ncrystal"
    fi
    .  $PTPATH/external/ncrystal/install/setup.sh
  fi


#install VecGeom

if [ ! -f $PTPATH/external/VecGeom/install/lib/libvecgeom.a ]; then
  read -r -p "Do you want to install VecGeom into $PTPATH/external? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    if [ ! -d $PTPATH/external ]; then
      mkdir $PTPATH/external
    fi
    cd $PTPATH/external
    if [ -d VecGeom ]; then
      rm -rf VecGeom
    fi
    git clone https://gitlab.com/xxcai1/VecGeom.git
    cd -
    mkdir $PTPATH/external/VecGeom/build && cd $PTPATH/external/VecGeom/build
    cmake  -DCMAKE_INSTALL_PREFIX=$PTPATH/external/VecGeom/install -DGDML=On -DUSE_NAVINDEX=On  ..
    make -j ${NUMCPU} && make install
    cd -
    echo "installed  VecGeom"
  else
    echo "Found VecGeom"
  fi
fi

if [ -f $PTPATH/src/python/Prompt/__init__.py ]; then
  export PYTHONPATH="$PTPATH/src/python:$PYTHONPATH"
  echo "Added prompt python module into path"
else
  echo "Can not find prompt python module"
fi


if [ ! -d $PTPATH/promptbin ]; then
  mkdir $PTPATH/promptbin
  cd $PTPATH/promptbin
  cmake ..
  make -j8
  cd -
fi


if [ -d $PTPATH/promptbin ]; then
  #export PATH="$PTPATH/resource/cxx/src:$PATH"
  #export PATH="$PTPATH/resource/src/bin:$PATH"
  export LD_LIBRARY_PATH="$PTPATH/promptbin:$LD_LIBRARY_PATH"
  export DYLD_LIBRARY_PATH="$PTPATH/promptbin:$DYLD_LIBRARY_PATH"
  export PATH="$PTPATH/promptbin/bin:$PTPATH/scripts:$PATH"
  echo "Added the promptbin directory into environment"
fi


if [ -f $PTPATH/ptvirenv/bin/activate ]; then
  . $PTPATH/ptvirenv/bin/activate
else
  read -r -p "Do you want install Prompt python virtual environment in $PTPATH/ptvirenv? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    python3 -m venv $PTPATH/ptvirenv
    . $PTPATH/ptvirenv/bin/activate
    pip install -r $PTPATH/requirement
  fi
fi


# ln -s
echo -e "Enjoy!"
