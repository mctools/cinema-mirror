#!/bin/bash

for i in {1..4}
do
  echo run $i
  prompt -n 1e7 -s $i -g PSD.gdml &
 done

i=5
prompt -n 1e7 -s $i -g PSD.gdml
