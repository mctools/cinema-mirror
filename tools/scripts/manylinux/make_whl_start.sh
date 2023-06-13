#!/bin/bash
#prerequest: install docker engine, see https://docs.docker.com/engine/install/ubuntu/ (note: linux distribution specific)
#run docker

# --set options for git repos
while getopts "hab:" option
  do
    case "${option}"
     in
      h) echo "-a               gitee"
         echo "-b               customized git repo"
         echo "-c               xxc"
         echo "other argument   gitlab"
         echo "no input         default (gitlab)"
         return 0;;
      a) PREFIX="https://ziyipan:cinema2022@gitee.com/ziyipan";;
      b) echo "Using repository from: "$1
         PREFIX=$1;;
      c) PREFIX="https://gitlab.com/xxcai1";;
      *) PREFIX="https://gitlab.com/cinema-developers";
    esac 
done
if (( $OPTIND == 1 )); then
  PREFIX="https://gitlab.com/cinema-developers"
fi 

chmod +x make_whl_main.sh
sudo docker run --env PREFIX=${PREFIX} -i -t \
-v `pwd`/../../../:/io quay.io/pypa/manylinux2014_x86_64 /io/tools/scripts/manylinux/make_whl_main.sh