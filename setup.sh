#!/bin/bash

#remove previously sourced env
function pixiu_prunepath() {
    P=$(IFS=:;for p in ${!1}; do [[ $p != ${2}* ]] && echo -n ":$p" || :; done)
    export $1=${P:1:99999}
}

if [ ! -z ${PIXIUPATH:-} ]; then
    #pixiu_prunepath PATH "$PIXIUPATH"
    #pixiu_prunepath LD_LIBRARY_PATH "$PIXIUPATH"
    #pixiu_prunepath DYLD_LIBRARY_PATH "$PIXIUPATH"
    pixiu_prunepath PYTHONPATH "$PIXIUPATH"
    echo "Cleaned up previously defined PiXiu enviorment"
fi

unset pixiu_prunepath

export PIXIUPATH="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f $PIXIUPATH/python/__init__.py ]; then
  export PYTHONPATH="$PIXIUPATH/python:$PYTHONPATH"
fi
