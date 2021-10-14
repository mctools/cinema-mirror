#!/bin/bash

#remove previously sourced env
function prunepath() {
    P=$(IFS=:;for p in ${!1}; do [[ $p != ${2}* ]] && echo -n ":$p" || :; done)
    export $1=${P:1:99999}
}

if [ ! -z ${PROMPTPATH:-} ]; then
    pixiu_prunepath PATH "$PROMPTPATH"
    pixiu_prunepath LD_LIBRARY_PATH "$PROMPTPATH"
    pixiu_prunepath DYLD_LIBRARY_PATH "$PROMPTPATH"
    pixiu_prunepath PYTHONPATH "$PROMPTPATH"
    echo "Cleaned up previously defined PiXiu enviorment"
fi

unset prunepath

export PROMPTPATH="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f $PROMPTPATH/src/python/PiXiu/__init__.py ]; then
  export PYTHONPATH="$PROMPTPATH/src/python:$PYTHONPATH"
  export PROMPTDATA="$PROMPTPATH/data"
  echo "Added PiXiu python module and data into path"
else
  echo "Can not find PiXiu python module"
fi

#if [ ! -d $PROMPTPATH/resource ]; then
  if [ ! -f $PROMPTPATH/resource//src/cxx/libprompt.so ]; then
    echo -e "C++ libPiXiu is not found. Slower python code can be used in low level calculation."
    read -r -p "Or, do you want install the accelarated C++ code in $PROMPTPATH/resource? [y/N] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      if [ ! -d $PROMPTPATH/resource ]; then
        mkdir $PROMPTPATH/resource
      fi
      cd $PROMPTPATH/resource
      cmake .. && make -j
      cd -
    else
      echo "Skipped the installation of libPiXiu"
    fi
  fi
#fi

if [ -f $PROMPTPATH/resource/src/cxx/libprompt.so ]; then
  #export PATH="$PROMPTPATH/resource/cxx/src:$PATH"
  #export PATH="$PROMPTPATH/resource/src/bin:$PATH"
  export LD_LIBRARY_PATH="$NCRYSTALDIR/resource/src/cxx:$LD_LIBRARY_PATH"
  export DYLD_LIBRARY_PATH="$NCRYSTALDIR/resource/src/cxx:$DYLD_LIBRARY_PATH"
  export PROMPTLIB=$PROMPTPATH/resource/src/cxx
  echo "Added C++ libPiXiu into the enviorment"
else
  echo "Warning: C++ lib is not found"
fi

if ! which pw.x > /dev/null; then
   echo -e "Warning: pw.x quantum-espresso is not found. Install it if phonon calculation is of your interest."
fi

if [ -f $PROMPTPATH/resource/pixiu/bin/activate ]; then
  . $PROMPTPATH/resource/pixiu/bin/activate
else
  read -r -p "Do you want install pixiu python virtual environment in $PROMPTPATH/resource? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    python3 -m venv $PROMPTPATH/resource/pixiu
    . $PROMPTPATH/resource/pixiu/bin/activate
    python -m pip install numpy
    python -m pip install scipy
    python -m pip install matplotlib
    python -m pip install h5py
    python -m pip install phonopy
    python -m pip install pyqt5
    python -m pip install vitables
  fi
fi

###install the standard solid-state pseudopotentials (SSSP) library from https://www.materialscloud.org/

if [ ! -f $PROMPTPATH/resource/sssp/sssp_efficiency.json ]; then
  read -r -p "Do you want to download the SSSP library into $PROMPTPATH/resource? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    mkdir $PROMPTPATH/resource/sssp
    wget https://www.materialscloud.org/discover/data/discover/sssp/downloads/sssp_precision.json -P $PROMPTPATH/resource/sssp
    wget https://www.materialscloud.org/discover/data/discover/sssp/downloads/sssp_efficiency.json -P $PROMPTPATH/resource/sssp

    mkdir $PROMPTPATH/resource/sssp/precision
    wget https://www.materialscloud.org/discover/data/discover/sssp/downloads/SSSP_precision_pseudos.tar.gz -P $PROMPTPATH/resource/sssp/precision
    cd $PROMPTPATH/resource/sssp/precision && tar -xzvf SSSP_precision_pseudos.tar.gz && cd -

    mkdir $PROMPTPATH/resource/sssp/efficiency
    wget https://www.materialscloud.org/discover/data/discover/sssp/downloads/SSSP_efficiency_pseudos.tar.gz -P $PROMPTPATH/resource/sssp/efficiency
    cd $PROMPTPATH/resource/sssp/efficiency && tar -xzvf SSSP_efficiency_pseudos.tar.gz --strip-components 1 && cd -
  fi
fi
export PROMPTSSSP=$PROMPTPATH/resource/sssp


echo -e "Enjoy!"
