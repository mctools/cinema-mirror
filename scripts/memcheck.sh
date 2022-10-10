#!/bin/bash

USAGE="Usage: $0 -x input_excutable -i args"

while getopts hi:x: option
do
case "${option}"
in
h) echo "$USAGE"
   exit 0;;
x) INPUT=${OPTARG};;
i) ARGS=${OPTARG};;
esac
done
valgrind --leak-check=full --show-leak-kinds=all -s ${INPUT} ${ARGS}
