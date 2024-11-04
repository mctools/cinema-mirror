#!/bin/bash

##### Input #####
INPUT='kds_bl9'    # McStas istrument name, without .instr
OutDir='mcstas' # Directory where execute simulation and store output files
N=1E7          # Number of particles to 
KDSourceLibPath=/home/panzy/project/ml/external/KDSource/install/
### End Input ###

# Execute McStas
rm -rf $OutDir
export LD_LIBRARY_PATH=$KDSourceLibPath/lib
cp $KDSourceLibPath/mcstas/contrib/KDSource.comp .
mcrun -c $INPUT --dir=$OutDir -n $N --format=NeXus
rm KDSource.comp $INPUT.c $INPUT.out
echo Simulation ended successfully