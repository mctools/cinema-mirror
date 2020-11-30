#!/bin/bash

export PIXIUPATH="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f $PIXIUPATH/python/__init__.py ]; then
    export PYTHONPATH="$PIXIUPATH/python:$PYTHONPATH"
fi
