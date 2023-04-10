#!/bin/bash

for i in {1..49}
do
  echo run $i
  prompt -n 4e8 -s $i -g total_scattering.gdml &
 done

i=50
prompt -n 4e8 -s $i -g total_scattering.gdml &
