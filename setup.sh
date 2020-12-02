#!/bin/bash

#remove previously sourced env
function pixiu_prunepath() {
    P=$(IFS=:;for p in ${!1}; do [[ $p != ${2}* ]] && echo -n ":$p" || :; done)
    export $1=${P:1:99999}
}

if [ ! -z ${PIXIUPATH:-} ]; then
    pixiu_prunepath PATH "$PIXIUPATH"
    pixiu_prunepath LD_LIBRARY_PATH "$PIXIUPATH"
    pixiu_prunepath DYLD_LIBRARY_PATH "$PIXIUPATH"
    pixiu_prunepath PYTHONPATH "$PIXIUPATH"
    echo "Cleaned up previously defined PiXiu enviorment"
fi

unset pixiu_prunepath

export PIXIUPATH="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f $PIXIUPATH/python/__init__.py ]; then
  export PYTHONPATH="$PIXIUPATH/python:$PYTHONPATH"
fi


if [ -f $PIXIUPATH/rundir/cxx/src/libPiXiu.so ]; then
  export PATH="$PIXIUPATH/rundir/cxx/src:$PATH"
  export PATH="$PIXIUPATH/rundir/src/bin:$PATH"
  export LD_LIBRARY_PATH="$NCRYSTALDIR/rundir/cxx/src:$LD_LIBRARY_PATH"
  export DYLD_LIBRARY_PATH="$NCRYSTALDIR/rundir/cxx/src:$DYLD_LIBRARY_PATH"
else
  echo "C++ lib is not found"
fi
