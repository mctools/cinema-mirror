#!/bin/bash

USAGE="Usage: $0 -x input_excutable -a args"

while getopts hx:a: option
do
case "${option}"
in
h) echo "$USAGE"
   exit 0;;
x) INPUT=${OPTARG};;
o) ARGS=${OPTARG};;
esac


valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes  ${INPUT} ${ARGS}

done
