#!/bin/bash

USAGE="Usage: $0 -i input_excutable -o output_file "

while getopts hi:o: option
do
case "${option}"
in
h) echo "$USAGE"
   exit 0;;
i) INPUT=${OPTARG};;
o) OUTPUT=${OPTARG};;
esac


valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes  ${INPUT}

done
