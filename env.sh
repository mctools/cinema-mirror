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
    *) echo "$USAGE"
      exit 0;;
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

#if ! hash pw.x 2>/dev/null
#then
#    echo "pw.x is required to continue the installation of Cinema"
#    return
#fi

. "$( dirname "${BASH_SOURCE[0]}" )"/install.sh

alias pull='git pull --recurse-submodules'
# ln -s
echo -e "Enjoy!"
