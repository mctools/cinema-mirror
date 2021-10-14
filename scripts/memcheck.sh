#!/bin/bash

USAGE="Usage: $0 -i input_excutable "

while getopts hi: option
do
case "${option}"
in
h) echo "$USAGE"
   exit 0;;
i) INPUT=${OPTARG};;
esac
done
valgrind --leak-check=full  ${INPUT}
